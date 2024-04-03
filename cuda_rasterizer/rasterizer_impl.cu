/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
/**
 * @brief 
 * 
 * @param P 
 * @param points_xy 
 * @param depths 
 * @param offsets 
 * @param gaussian_keys_unsorted [tile|depth]-key
 * @param gaussian_values_unsorted [idx]-value
 * @param radii 
 * @param grid 
 * @return __global__ 
 */
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	// 保证高斯点是可见
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// 找到高斯idx在buffer中的偏移量，作为存储高斯key、value数组的起始位置
		// offsets为前缀和数组，取出idx前一项的值
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// 获取当前高斯在2D的矩形边界
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// 遍历高斯投影矩形覆盖的每个tile
		// 对每个tile和高斯对应的depth组成[tile|depth] key，value为线程idx对应高斯id
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				// 直接将float转为uint32_t，在当前float标准下保证了序贯性
				// 跨平台时可能会有问题
				key |= *((uint32_t*)&depths[idx]);
				// 对于每个thread off是独立的
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
/**
 * @brief 确定每个tile内待渲染的高斯id范围
 * 
 * @param L [in]
 * @param point_list_keys [in]
 * @param ranges [out] 大小为tile数量，
 * 					   每个对象记录对应tile的起始和结束高斯id
 * 					   [起始, 结束)
 * @return  
 */
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	// 当前线程id对应1个高斯实例
	auto idx = cg::this_grid().thread_rank();
	// idx超出待渲染高斯实例的线程不处理
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	// 取出高斯id在排序后list中对应key[tile | depth]
	uint64_t key = point_list_keys[idx];
	// 取出key高32位对应的tile id
	uint32_t currtile = key >> 32;
	// 第1个高斯，将当前tile的起始位置设为0
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		// 取出前一个id高斯对应的tile id
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 不是同一个tile，当前高斯和前一个高斯在不同的tile
		if (currtile != prevtile)
		{
			// 更新前一tile的最后一个高斯id
			ranges[prevtile].y = idx;
			// 当前tile的起始高斯id 
			ranges[currtile].x = idx;
		}
	}
	// 最后1个高斯，将当前tile的结束位置设为L
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}
/**
 * @brief 从内存块中获取数据并将结果处理存储在BinningState对象中
 * 
 * @param chunk 
 * @param P 
 * @return CudaRasterizer::BinningState 
 */
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	int* radii,
	bool debug)
{
	// 计算fx fy
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 初始化geometryBuffer
	// 获取geometryBuffer的大小
	size_t chunk_size = required<GeometryState>(P);
	// 获取geometryBuffer的指针
	char* chunkptr = geometryBuffer(chunk_size);
	// 获取GeometryState对象
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}
	// tile grid的大小 (width向上取整，block_x整数倍，height向上取整，block_y整数倍)
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// block的大小16x16 pixel
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// 根据训练图像大小初始化imageBuffer
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 执行preprocess(CHECK_CUDA检查是否成功运行)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// tiles_touched按照线程idx保存对应高斯点在2D屏幕上覆盖的tile数
	// 计算tiles_touched前缀和，保存在point_offsets中
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// 取出要渲染的高斯实例，分配辅助buffer
	int num_rendered;
	// 取出前缀和数组的最后一位置(geomState.point_offsets + P - 1)
	// int大小数据，表示全部高斯点覆盖的tile总数
	// 从gpu拷贝到cpu给num_rendered，也是要渲染的高斯实例数
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	// 根据num_rendered重新分配binningBuffer(初始化时已经在gpu上分配内存)
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 对每个待渲染高斯实例生成[tile|depth]-key [idx]-value
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)
	// 找到tile的最高有效位MSB
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 对[0, 32+bit]范围内的key-[tile|depth]和对应的高斯实例idx进行排序
	// 基数排序默认为升序
	/**
	 * @brief cub::DeviceRadixSort::SortPairs 基数排序
	 * @param[in] list_sorting_space 临时空间
	 * @param[in] sorting_size 临时空间大小
	 * @param[in] point_list_keys_unsorted 输入key
	 * @param[out] point_list_keys 输出key
	 * @param[in] point_list_unsorted 输入value
	 * @param[out] point_list 输出value
	 * @param[in] num_rendered 输入key的数量
	 * @param[in] begin_bit 开始比特位
	 * @param[in] end_bit 结束比特位
	 * 
	 */
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// 将imgState.ranges的(tile大小 * uint2)范围内存初始化为0
	// 大小为tile数量，每个对象记录对应tile的起始和结束高斯实例id
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	// 待渲染的高斯数大于0
	// 确定每个tile对应待渲染高斯id的范围
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	// 按tile并行渲染
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	// 取出高斯的深度
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		depth_ptr,
		geomState.conic_opacity,
		// 输出，记录T的终值
		imgState.accum_alpha,
		// 输出，记录每个像素贡献的高斯数
		imgState.n_contrib,
		background,
		out_color,
		out_depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	// 输入变量，pytorch自动计算的梯度
	const float* dL_dpix,
	const float* dL_ddepths,
	// 输出变量，cuda计算的梯度
	// render返回
	float* dL_dmean2D,
	// render返回
	float* dL_dconic,
	// render返回
	float* dL_dopacity,
	// render返回
	float* dL_dcolor,
	// preprocess返回
	float* dL_dmean3D,
	// preprocess返回
	float* dL_dcov3D,
	// preprocess返回
	float* dL_dsh,
	// preprocess返回
	float* dL_dscale,
	// preprocess返回
	float* dL_drot,
	bool debug)
{
	// 从传入变量获取对象
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 以上部分与forward操作相同

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		// 输入，高斯的深度
		depth_ptr,
		// 输入，记录forward中T的终值
		imgState.accum_alpha,
		imgState.n_contrib,
		// 输入变量
		dL_dpix,
		dL_ddepths,
		// render 输出4个梯度
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		// render计算出，输入preprocess
		(float3*)dL_dmean2D,
		// render计算出，输入preprocess
		dL_dconic,
		// 输出
		(glm::vec3*)dL_dmean3D,
		// render计算出，输入preprocess
		dL_dcolor,
		// 输出
		dL_dcov3D,
		// 输出
		dL_dsh,
		// 输出
		(glm::vec3*)dL_dscale,
		// 输出
		(glm::vec4*)dL_drot), debug)
}