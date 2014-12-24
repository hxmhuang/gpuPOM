#ifndef CADVANCE_GPU_KERNEL_H
#define CADVANCE_GPU_KERNEL_H

/*
__global__ void
surface_forcing_gpu_kernel_0(float *wusurf, float *wvsurf, 
						     float *e_atmos, float *swrad,
						     float *vfluxf, float *w,
						     float *wtsurf, float *wssurf, 
						     float *t,  float *s,
						     float tbias, float sbias, 
						     int jm, int im);
*/

__global__ void
surface_forcing_gpu_kernel_0(float * __restrict__ e_atmos, 
							 float * __restrict__ swrad,
						     const float * __restrict__ vfluxf, 
							 float * __restrict__ w,
						     int jm, int im);


__global__ void
momentum3d_gpu_kernel_0(float * __restrict__ aam, 
						float aam_init,
						int kb, int jm, int im);


__global__ void
momentum3d_gpu_kernel_1(float * __restrict__ aam, 
						const float * __restrict__ aamfrz,
					    const float * __restrict__ u, 
						const float * __restrict__ v,
						const float * __restrict__ dx, 
						const float * __restrict__ dy,
						float horcon,
						int kb, int jm, int im);

__global__ void
mode_interaction_gpu_kernel_0(
				float * __restrict__ adx2d, 
				const float * __restrict__ advx,
				float * __restrict__ ady2d, 
				const float * __restrict__ advy,
				float * __restrict__ drx2d, 
				const float * __restrict__ drhox,
				float * __restrict__ dry2d, 
				const float * __restrict__ drhoy,
				float * __restrict__ aam2d, 
				const float * __restrict__ aam,
				const float * __restrict__ dz, 
				int kb, int jm, int im);


__global__ void
mode_interaction_gpu_kernel_1(float * __restrict__ adx2d, 
							  float * __restrict__ ady2d,
							  const float * __restrict__ advua, 
							  const float * __restrict__ advva,
							  int jm, int im);

__global__ void
mode_interaction_gpu_kernel_2(float * __restrict__ egf, 
							  const float * __restrict__ el,
							  float * __restrict__ utf, 
							  float * __restrict__ vtf, 
							  const float * __restrict__ ua, 
							  const float * __restrict__ va,
							  const float * __restrict__ d,
							  float ispi, float isp2i,
							  int jm, int im);

__global__ void
mode_external_gpu_kernel_0(float * __restrict__ fluxua, 
						   float * __restrict__ fluxva,
						   const float * __restrict__ ua, 
						   const float * __restrict__ va,
						   const float * __restrict__ d, 
						   const float * __restrict__ dx, 
						   const float * __restrict__ dy,
						   int jm, int im);

__global__ void
mode_external_gpu_kernel_1(const float * __restrict__ fluxua, 
						   const float * __restrict__ fluxva,
						   float * __restrict__ elf, 
						   const float * __restrict__ elb,
						   const float * __restrict__ vfluxf, 
						   const float * __restrict__ art, 
						   float dte2, int jm, int im);

__global__ void
mode_external_gpu_kernel_2(float * __restrict__ uaf, 
						   const float * __restrict__ ua, 
						   const float * __restrict__ uab, 
						   float * __restrict__ vaf, 
						   const float * __restrict__ va, 
						   const float * __restrict__ vab,
						   const float * __restrict__ elf, 
						   const float * __restrict__ el, 
						   const float * __restrict__ elb,
						   const float * __restrict__ adx2d, 
						   const float * __restrict__ ady2d, 
						   const float * __restrict__ advua, 
						   const float * __restrict__ advva, 
						   const float * __restrict__ drx2d, 
						   const float * __restrict__ dry2d, 
						   const float * __restrict__ wusurf, 
						   const float * __restrict__ wvsurf, 
						   const float * __restrict__ wubot, 
						   const float * __restrict__ wvbot,
						   const float * __restrict__ e_atmos, 
						   const float * __restrict__ d,
						   const float * __restrict__ dx, 
						   const float * __restrict__ dy,
						   const float * __restrict__ aru, 
						   const float * __restrict__ arv, 
						   const float * __restrict__ cor, 
						   const float * __restrict__ h,
						   float grav, float alpha, float dte,
						   int jm, int im);

__global__ void
mode_external_gpu_kernel_3(float * __restrict__ etf, 
						   float * __restrict__ d,
						   float * __restrict__ ua, 
						   float * __restrict__ uab, 
						   const float * __restrict__ uaf,
						   float * __restrict__ va, 
						   float * __restrict__ vab, 
						   const float * __restrict__ vaf,
						   float * __restrict__ el, 
						   float * __restrict__ elb, 
						   const float * __restrict__ elf,
						   const float * __restrict__ fsm, 
						   const float * __restrict__ h,
						   int iext, int isplit, float smoth,
						   int jm, int im);

__global__ void
mode_external_gpu_kernel_4(const float * __restrict__ d,
						   float * __restrict__ utf, 
						   float * __restrict__ vtf, 
						   float * __restrict__ egf, 
						   const float * __restrict__ ua, 
						   const float * __restrict__ va, 
						   const float * __restrict__ el, 
						   float ispi, float isp2i,
						   int iext, int isplit,
						   int jm, int im);

__global__ void
mode_internal_gpu_kernel_0(float * __restrict__ u, 
						   float * __restrict__ v, 
						   const float * __restrict__ utb, 
						   const float * __restrict__ utf, 
						   const float * __restrict__ vtb, 
						   const float * __restrict__ vtf, 
						   const float * __restrict__ dt,
						   const float * __restrict__ dz, 
						   int kb, int jm, int im);

__global__ void 
mode_internal_gpu_kernel_1(float * __restrict__ q2, 
						   float * __restrict__ q2b,
						   float * __restrict__ q2l, 
						   float * __restrict__ q2lb,
						   const float * __restrict__ uf, 
						   const float * __restrict__ vf, 
						   float smoth,
						   int kb, int jm, int im);

__global__ void
mode_internal_gpu_kernel_2(float * __restrict__ tb, 
						   float * __restrict__ t, 
						   float * __restrict__ uf,
						   float * __restrict__ sb, 
						   float * __restrict__ s, 
						   float * __restrict__ vf,
						   float smoth, 
						   int kb, int jm, int im);

__global__ void
mode_internal_gpu_kernel_3(float * __restrict__ ub, 
						   float * __restrict__ u, 
						   const float * __restrict__ uf, 
						   float * __restrict__ vb, 
						   float * __restrict__ v, 
						   const float * __restrict__ vf, 
						   float *tps, 
						   const float * __restrict__ dz, 
						   float smoth,
						   int kb, int jm, int im);

__global__ void
mode_internal_gpu_kernel_4(float * __restrict__ egb, 
						   const float * __restrict__ egf, 
						   float * __restrict__ etb, 
						   float * __restrict__ et, 
						   const float * __restrict__ etf,
						   float * __restrict__ utb, 
						   const float * __restrict__ utf,
						   float * __restrict__ vtb, 
						   const float * __restrict__ vtf,
						   float * __restrict__ vfluxb,
						   const float * __restrict__ vfluxf,
						   float * __restrict__ dt, 
						   const float * __restrict__ h,
						   int jm, int im);

__global__ void
store_mean_gpu_kernel_0(float * __restrict__ uab_mean, 
						float * __restrict__ vab_mean,
						float * __restrict__ elb_mean,
						float * __restrict__ wusurf_mean, 
						float * __restrict__ wvsurf_mean,
						float * __restrict__ wtsurf_mean,
						float * __restrict__ wssurf_mean,
						float * __restrict__ u, 
						float * __restrict__ v, 
						float * __restrict__ w,
						float * __restrict__ aam,
						float * __restrict__ u_mean,
						float * __restrict__ v_mean,
						float * __restrict__ w_mean,
						float * __restrict__ t_mean,
						float * __restrict__ s_mean,
						float * __restrict__ rho_mean,
						float * __restrict__ kh_mean,
						float * __restrict__ km_mean,
						const float * __restrict__ cor,
						const float * __restrict__ aam2d,
						const float * __restrict__ e_atmos,
						const float * __restrict__ uab, 
						const float * __restrict__ vab, 
						const float * __restrict__ elb,
						const float * __restrict__ wusurf, 
						const float * __restrict__ wvsurf,
						const float * __restrict__ wtsurf, 
						const float * __restrict__ wssurf,
						const float * __restrict__ wubot, 
						const float * __restrict__ wvbot,
						const float * __restrict__ ustks, 
						const float * __restrict__ cbc,
						const float * __restrict__ vstks,
						const float * __restrict__ t, 
						const float * __restrict__ s,
						const float * __restrict__ rho, 
						const float * __restrict__ kh,
						int mode, int kb, int jm, int im);


__global__ void
store_surf_mean_gpu_kernel_0(
						float * __restrict__ usrf_mean, 
						float * __restrict__ vsrf_mean,
						float * __restrict__ elb,
						float * __restrict__ elsrf_mean,
						float * __restrict__ uwsrf_mean, 
						float * __restrict__ vwsrf_mean,
						float * __restrict__ utf_mean, 
						float * __restrict__ vtf_mean,
						float * __restrict__ xstks_mean, 
						float * __restrict__ ystks_mean,
						float * __restrict__ celg_mean,
						float * __restrict__ ctsurf_mean,
						float * __restrict__ ctbot_mean,
						float * __restrict__ cpvf_mean,
						float * __restrict__ cjbar_mean,
						float * __restrict__ cadv_mean,
						float * __restrict__ cten_mean,

						const float * __restrict__ uab, 
						const float * __restrict__ vab,
						const float * __restrict__ u, 
						const float * __restrict__ v,
						const float * __restrict__ cor, 
						const float * __restrict__ uwsrf, 
						const float * __restrict__ vwsrf,
						const float * __restrict__ wusurf, 
						const float * __restrict__ wvsurf,
						const float * __restrict__ utf, 
						const float * __restrict__ vtf,
						const float * __restrict__ xstks, 
						const float * __restrict__ ystks,
						const float * __restrict__ celg, 
						const float * __restrict__ ctsurf,
						const float * __restrict__ ctbot,
						const float * __restrict__ cpvf,
						const float * __restrict__ cjbar, 
						const float * __restrict__ cadv, 
						const float * __restrict__ cten,
						int mode, int calc_wind, int calc_vort,
						int kb, int jm, int im);


#endif
