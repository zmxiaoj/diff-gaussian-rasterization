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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	// focal_x focal_y
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	// 输入梯度
	const float* dL_dconics,
	// 输入梯度，render计算出，深度损失关于相机坐标系下z的梯度
	const float* dL_dviewz,
	// 输出梯度
	float3* dL_dmeans,
	// 输出梯度
	float* dL_dcov)
{
	// 取出当前thread的id
	auto idx = cg::this_grid().thread_rank();
	// 每个thread处理一个高斯，跳过超出idx或半径为0的高斯
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	// 获取当前thread处理高斯的3D Cov矩阵
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	// 取出当前thread处理高斯的均值
	float3 mean = means[idx];
	// 取出当前thread处理高斯的2D Cov
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	// 将高斯中心从世界坐标系转换到相机坐标系
	float3 t = transformPoint4x3(mean, view_matrix);
	
	// 计算水平和垂直方向的视场角tan范围，考虑有的高斯中心超出视场范围但是半径很大
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	// 分别计算x和y关于z的归一化坐标
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// 将x和y的大小限制在视场范围内，并恢复深度
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	// 标记梯度是否有效，超出范围的梯度无效
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	// 矩阵按照列优先进行存储，J^T
	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 矩阵按照列优先进行存储，W^T 
	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	// 计算投影后的Cov2D = J * W * Cov3D * W^T * J^T
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	// 2D Cov的上三角
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	// 2D Cov的det
	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	// 2D Cov逆矩阵
	// [d e; e f] = 1/det*[c -b; -b a]
	// d=c/(ac-b^2) e=-b/(ac-b^2) f=a/(ac-b^2) 
	// dd_da=-c^2/(ac-b^2)^2 dd_db=2bc/(ac-b^2)^2        dd_dc=-b^2/(ac-b^2)^2
	// de_da=bc/(ac-b^2)^2   de_db=(-ac-b^2)/(ac-b^2)^2 de_dc=ab/(ac-b^2)^2
	// df_da=-b^2/(ac-b^2)^2 df_db=2ab/(ac-b^2)^2       df_dc=-a^2/(ac-b^2)^2

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		// 矩阵对称部分*2，e的部分*2
		// dL_da = dL_dd * dd_da + 2 * dL_de * de_da + dL_df * df_da
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		// dL_dc = dL_dd * dd_dc + 2 * dL_de * de_dc + dL_df * df_dc
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		// dL_db = 2 * (dL_dd * dd_db + 2 * dL_de * de_db + dL_df * df_db)
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// Cov2D = [a b; c d]
		// Cov3D = [0 1 2; 1 3 4; 2 4 5]
		// T = [0 1 2; 3 4 5; 6 7 8]
		// a=C0*T0^2  +2C1*T0*T1        +2C2*T0*T2        +C3*T1^2  +2C4*T1*T2        +C5*T2^2
		// b=C0*T0*T3 +C1*(T0*T4+T1*T3) +C2*(T0*T5+T2*T3) +C3*T1*T4 +C4*(T1*T5+T2*T4) +C5*T2*T5
		// c=C0*T3^2  +2C1*T3*T4        +2C2*T3*T5        +C3*T4^2  +2C4*T4*T5        +C5*T5^2
		// dL_dC0 = dL_da * da_dC0 + dL_db * db_dC0 + dL_dc * dc_dC0
		//        = T0^2 * dL_da + T0 * T3 * dL_db + T3^2 * dL_dc
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		// dL_dC3 = dL_da * da_dC3 + dL_db * db_dC3 + dL_dc * dc_dC3
		//        = T1^2 * dL_da + T1 * T4 * dL_db + T4^2 * dL_dc
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		// dL_dC5 = dL_da * da_dC5 + dL_db * db_dC5 + dL_dc * dc_dC5
		//        = T2^2 * dL_da + T2 * T5 * dL_db + T5^2 * dL_dc
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// dL_dC1 = dL_da * da_dC1 + dL_db * db_dC1 + dL_dc * dc_dC1
		//        = 2 * T0 * T1 * dL_da + (T0 * T4 + T1 * T3) * dL_db + 2 * T3 * T4 * dL_dc
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		// dL_dC2 = dL_da * da_dC2 + dL_db * db_dC2 + dL_dc * dc_dC2
		//        = 2 * T0 * T2 * dL_da + (T0 * T5 + T2 * T3) * dL_db + 2 * T3 * T5 * dL_dc
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		// dL_dC4 = dL_da * da_dC4 + dL_db * db_dC4 + dL_dc * dc_dC4
		//        = 2 * T1 * T2 * dL_da + (T1 * T5 + T2 * T4) * dL_db + 2 * T4 * T5 * dL_dc
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	// a=C0*T0^2  +2C1*T0*T1        +2C2*T0*T2        +C3*T1^2  +2C4*T1*T2        +C5*T2^2
	// b=C0*T0*T3 +C1*(T0*T4+T1*T3) +C2*(T0*T5+T2*T3) +C3*T1*T4 +C4*(T1*T5+T2*T4) +C5*T2*T5
	// c=C0*T3^2  +2C1*T3*T4        +2C2*T3*T5        +C3*T4^2  +2C4*T4*T5        +C5*T5^2
	// Cov 2D与T的第3行无关，只计算损失关于T前两行的梯度
	// dL_dT0 = 2 * (T0 * C0 + 2 * T1 * C1 + 2 * T2 * C2) * dL_da + (T3 * C0 + T4 * C1 + T5 * C2) * dL_db
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	// dL_dT1 = 2 * (T0 * C1 + 2 * T1 * C3 + 2 * T2 * C4) * dL_da + (T3 * C1 + T4 * C3 + T5 * C4) * dL_db
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	// dL_dT2 = 2 * (T0 * C2 + 2 * T1 * C4 + 2 * T2 * C5) * dL_da + (T3 * C2 + T4 * C4 + T5 * C5) * dL_db
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	// dL_dT3 = (T0 * C0 + 2 * T1 * C1 + 2 * T2 * C2) * dL_db + 2 * (T3 * C0 + T4 * C1 + T5 * C2) * dL_dc
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	// dL_dT4 = (T0 * C1 + 2 * T1 * C3 + 2 * T2 * C4) * dL_db + 2 * (T3 * C1 + T4 * C3 + T5 * C4) * dL_dc
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	// dL_dT5 = (T0 * C2 + 2 * T1 * C4 + 2 * T2 * C5) * dL_db + 2 * (T3 * C2 + T4 * C4 + T5 * C5) * dL_dc
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J => 实际上是T^T = W^T * J^T => T = J * W
	// T = [W0 * J0 + W3 * J1 + W6 * J2; W1 * J0 + W4 * J1 + W7 * J2; W2 * J0 + W5 * J1 + W8 * J2]
	//   = [W0 * J3 + W3 * J4 + W6 * J5; W1 * J3 + W4 * J4 + W7 * J5; W2 * J3 + W5 * J4 + W8 * J5]
	//   = [W0 * J6 + W3 * J7 + W6 * J8; W1 * J6 + W4 * J7 + W7 * J8; W2 * J6 + W5 * J7 + W8 * J8]
	// 只关心T的前两行
	// J1 = 0, J3 = 0，损失关于J1 J3的梯度为0
	// dL_dJ0 = dL_dT0 * dT0_dJ0 + dL_dT1 * dT1_dJ0 + dL_dT2 * dT2_dJ0
	//        = W0 * dL_dT0 + W1 * dL_dT1 + W2 * dL_dT2
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	// dL_dJ2 = dL_dT0 * dT0_dJ2 + dL_dT1 * dT1_dJ2 + dL_dT2 * dT2_dJ2
	//        = W6 * dL_dT0 + W7 * dL_dT1 + W8 * dL_dT2
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	// dL_dJ4 = dL_dT3 * dT3_dJ4 + dL_dT4 * dT4_dJ4 + dL_dT5 * dT5_dJ4
	//        = W3 * dL_dT3 + W4 * dL_dT4 + W5 * dL_dT5
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	// dL_dJ5 = dL_dT3 * dT3_dJ5 + dL_dT4 * dT4_dJ5 + dL_dT5 * dT5_dJ5
	//        = W6 * dL_dT3 + W7 * dL_dT4 + W8 * dL_dT5
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// h_x = focal_x, h_y = focal_y
	// J0 = fx / z, J1 = 0, J2 = -fx * x / z^2
	// J3 = 0, J4 = fy / z, J5 = -fy * y / z^2
	// dJ0_dx = 0, dJ2_dx = -fx / z^2, dJ4_dx = 0, dJ5_dx = 0
	// dJ0_dy = 0, dJ2_dy = 0, dJ4_dy = 0, dJ5_dy = -fy / z^2
	// dJ0_dz = -fx / z^2, dJ2_dz = 2fx * x / z^3, dJ4_dz = -fy / z^2, dJ5_dz = 2fy * y / z^3
	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	// dL_dtx = dL_dJ0 * dJ0_dx + dL_dJ2 * dJ2_dx + dL_dJ4 * dJ4_dx + dL_dJ5 * dJ5_dx
	//        = -fx / z^2 * dL_dJ2
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	// dL_dty = dL_dJ0 * dJ0_dy + dL_dJ2 * dJ2_dy + dL_dJ4 * dJ4_dy + dL_dJ5 * dJ5_dy
	//        = -fy / z^2 * dL_dJ5 
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	// dL_dtz = dL_dJ0 * dJ0_dz + dL_dJ2 * dJ2_dz + dL_dJ4 * dJ4_dz + dL_dJ5 * dJ5_dz
	//        = -fx / z^2 * dL_dJ0 + 2fx * x / z^3 * dL_dJ2 - fy / z^2 * dL_dJ4 + 2fy * y / z^3 * dL_dJ5
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// 获取深度图计算出损失关于当前idx高斯在相机坐标系下z的梯度
	dL_dtz += dL_dviewz[idx];

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	// R_w2c * t_w + t_w2c = t_c
	// x_c = r0 * x_w + r1 * y_w + r2 * z_w + t0
	// y_c = r3 * x_w + r4 * y_w + r5 * z_w + t1
	// z_c = r6 * x_w + r7 * y_w + r8 * z_w + t2
	// dL_dtw = dL_dtc * dtc_dtw
	// dL_dxw = dL_dxc * dxc_dxw + dL_dyc * dyc_dxw + dL_dzc * dzc_dxw
	//        = r0 * dL_dxc + r3 * dL_dyc + r6 * dL_dzc
	// dL_dyw = dL_dxc * dxc_dyw + dL_dyc * dyc_dyw + dL_dzc * dzc_dyw
	//        = r1 * dL_dxc + r4 * dL_dyc + r7 * dL_dzc
	// dL_dzw = dL_dxc * dxc_dzw + dL_dyc * dyc_dzw + dL_dzc * dzc_dzw
	//        = r2 * dL_dxc + r5 * dL_dyc + r8 * dL_dzc
	// dL_dtw = [r0 r3 r6; r1 r4 r7; r2 r5 r8] * dL_dtc
	//        = R_w2c^T * dL_dtc
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	// full_proj_transform
	const float* proj,
	const glm::vec3* campos,
	// 输入，来自Backward::render
	const float3* dL_dmean2D,
	// 输出，
	glm::vec3* dL_dmeans,
	// 输入，来自Backward::render
	float* dL_dcolor,
	// 输入，来自computeCov2DCUDA
	float* dL_dcov3D,
	// 输出，
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	// 输入，记录高斯的深度
	const float* __restrict__ depths,
	// 输入，记录forward中T的终值
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	// 输入梯度
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_depth,
	// 输入，深度图每个像素对应的高斯id
	const int* __restrict__ depth_idx,
	// 输出4个梯度
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	// 深度损失关于高斯在相机坐标系下z的梯度
	float* __restrict__ dL_dviewz
	)
{
	// We rasterize again. Compute necessary block info.
	// 再次光栅化
	// 计算block相关信息
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	// 分配BLOCK_SIZE的共享内存，保存id、xy坐标、2D协方差矩阵的逆+不透明度、颜色
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	// 相比forward增加了保存3通道颜色的共享内存
	__shared__ float collected_colors[C * BLOCK_SIZE];
	// 增加保存渲染深度图的共享内存
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	// 在forward中保存了T的最终值，即所有(1 - alpha)因子的乘积
	// 每个thread取出合法像素(在图像范围内)对应的终值T 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	// tile对应的待处理高斯总数
	uint32_t contributor = toDo;
	// 取出forward中的最后一个高斯id
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	// 初始化辅助变量，记录从后向前累积的颜色
	float accum_rec[C] = { 0 };
	// 取出损失关于像素颜色3通道的梯度
	float dL_dpixel[C];
	// 初始化辅助变量，记录从后向前累积的深度
	float accum_depth_rec = 0;
	// 取出损失关于深度的梯度
	float dL_dpixel_depth = 0;
	// 记录深度图中像素对应的高斯id
	int depth_id;
	// 对于图像范围内的thread
	if (inside)
	{
		// 取出thread对应像素 损失关于颜色3通道对应的梯度
		for (int i = 0; i < C; i++)
			// dL_dpixels[i * H * W + 0], dL_dpixels[i * H * W + 1], ..., dL_dpixels[i * H * W + W*H-1]
			// 每行代表一张图像像素，每列代表一个通道
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixels_depth[pix_id];

		// 根据像素id取出对应的深度图中的高斯id
		depth_id = depth_idx[pix_id];
	}
	// 初始化变量，记录上一个高斯的alpha和3通道颜色
	float last_alpha = 0;
	float last_color[C] = { 0 };
	// // 记录上一个高斯的深度
	// float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	// 像素坐标关于NDC坐标的梯度
	// 对应视口变换的过程
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	// 遍历全部高斯点
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// 将辅助数据加载到共享内存中，从后向前加载

		// 区别于forward，先进行一次block内thread同步
		// 因为是从后向前遍历，不同pixel的起点不同，需要同步
		block.sync();
		// 当前线程处理的进度(从后往前)
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		// 在range范围内
		if (range.x + progress < range.y)
		{
			// thread从后往前(按照深度)取出高斯id
			const int coll_id = point_list[range.y - progress - 1];
			// 更新共享内存记录高斯相关变量
			// 共享内存中高斯属性是连续的从后往前排列(与forward中共享内存排列相反)
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			// 3通道的颜色分别连续排列在共享内存中
			// C_ch1_0 C_ch1_1 ... C_ch1_BLOCK_SIZE-1 
			// C_ch2_0 C_ch2_1 ... C_ch2_BLOCK_SIZE-1 
			// ... 
			// C_chC_0 C_chC_1 ... C_chC_BLOCK_SIZE-1
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			// 从后向前，取出深度
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		// 每个thread并行遍历共享内存中的高斯
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			// 从后向前处理高斯，小于last_contributor标记的开始
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			// 取出当前高斯的2D坐标
			const float2 xy = collected_xy[j];
			// 2D投影空间中像素坐标和gauss中心坐标的差
			// block内每个thread对应的pixel关于d不同
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			// 前3维 2D协方差矩阵逆的上三角，第4维 不透明度\alpha
			// 2D Cov [a b; b c]
			// 逆矩阵[d e; e f] + opacity
			const float4 con_o = collected_conic_opacity[j];
			// 2D高斯
			// -1/2 * [dx dy]^T * [cov1 cov2; cov2 cov3] * [dx dy]
			// = -1/2 * (cov1 * dx^2 + cov3 * dy^2 + 2 * cov2 * dx * dy)
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// block内每个pixle(thread)对相同高斯计算得到的G不同
			const float G = exp(power);
			// 验证Numerical-Stability
			const float alpha = min(0.99f, con_o.w * G);
			// \alpha小于阈值的高斯被跳过
			if (alpha < 1.0f / 255.0f)
				continue;

			// 当前位置对应的T_i(透射度累积到\alpha_{i-1})
			T = T / (1.f - alpha);
			// thread对应像素关于高斯颜色的梯度
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			// 损失关于alpha的梯度
			float dL_dalpha = 0.0f;
			// 当前处理的高斯id
			const int global_id = collected_id[j];
			// 遍历颜色通道
			for (int ch = 0; ch < C; ch++)
			{
				// thread取出当前pixel当前通道的颜色
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				// 从后向前，对颜色进行alpha composition
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				// 更新上一次颜色，用于下一次迭代
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.

				// dL_dcolors[global_id * C + 0], dL_dcolors[global_id * C + 1], ..., dL_dcolors[global_id * C + C-1]
				// 存储形式，每行表示一个高斯颜色，每列表示一个颜色通道				
				// 更新损失关于当前高斯颜色通道的梯度
				// 多个pixel关于同一个高斯颜色的梯度进行求和，需要原子操作
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			// 使用alpha-blending进行深度渲染时，需要考虑的深度损失
			// // thread取出当前pixel对应的渲染深度
			// const float c_d = collected_depths[j];
			// accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			// last_depth = c_d;
			// // 设计深度损失时增加这一项
			// dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			
			// median深度渲染时，需要考虑的深度损失
			// 如果当前高斯id对应深度图中像素的高斯id
			if (inside && global_id == depth_id)
			{
				// 计算深度损失
				// 原子更新
				atomicAdd(&(dL_dviewz[global_id]), dL_dpixel_depth);
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			// 更新上一次alpha，用于下一次迭代
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			// forward中，最终颜色 = 高斯颜色 + T * 背景颜色
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			// T_final为forward保存下T的最终值
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			//  G = exp(-1/2 * (cov1 * dx^2 + cov3 * dy^2 + 2 * cov2 * dx * dy))
			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			// 损失关于NDC坐标系上的像素坐标的梯度
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// 关于2D Cov逆矩阵 [d, e; e, f] 的梯度
			// .x .y .z(未使用) .w
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			// 关于e的梯度包括对称两部分，后续计算时需要注意计算两次
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			// 原子操作，将不同pixel的损失关于同一个高斯的不透明度梯度进行求和
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	// 输入梯度，render计算出
	const float3* dL_dmean2D,
	// 输入梯度，render计算出，对应dL_dconic2D
	const float* dL_dconic,
	// 输入梯度，render计算出，深度损失关于相机坐标系下z的梯度
	const float* dL_dviewz,
	// 输出梯度，computeCov2DCUDA
	glm::vec3* dL_dmean3D,
	// 输入梯度，render计算出
	float* dL_dcolor,
	// 输出梯度，computeCov2DCUDA
	float* dL_dcov3D,
	// 输出梯度，preprocessCUDA
	float* dL_dsh,
	// 输出梯度，preprocessCUDA
	glm::vec3* dL_dscale,
	// 输出梯度，preprocessCUDA
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		// 输入梯度，render计算出
		dL_dconic,
		// 输入梯度，render计算出，深度损失关于相机坐标系下z的梯度
		dL_dviewz,
		// 输出梯度
		(float3*)dL_dmean3D,
		// 输出梯度
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		// render计算出，输入梯度
		(float3*)dL_dmean2D,
		// computeCov2DCUDA计算出，输入梯度
		(glm::vec3*)dL_dmean3D,
		// render计算出，输入梯度
		dL_dcolor,
		// computeCov2DCUDA计算出，输入梯度
		dL_dcov3D,
		// 输出梯度
		dL_dsh,
		// 输出梯度
		dL_dscale,
		// 输出梯度
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	// 输入，背景颜色
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	// 输入，记录forward中T的终值
	const float* final_Ts,
	const uint32_t* n_contrib,
	// 输入梯度，损失关于rgb图的梯度，pytorch计算得到
	const float* dL_dpixels,
	// 输入梯度，损失关于深度图的梯度，pytorch计算得到
	const float* dL_dpixels_depth,
	const int* depth_idx,
	// 输出4个梯度
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	// 深度损失关于高斯在相机坐标系下z的梯度
	float*	dL_dviewz
	)
{
	// grid block_size(16x16)整倍数
	// block 16x16
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		// 输入，背景颜色
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		// 输入梯度
		dL_dpixels,
		dL_dpixels_depth,
		// 输入深度图每个像素对应的高斯id
		depth_idx,
		// 输出4个梯度
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		// 深度损失关于高斯在相机坐标系下z的梯度
		dL_dviewz
		);
}