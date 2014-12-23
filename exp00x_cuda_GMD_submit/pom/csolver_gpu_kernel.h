#ifndef CSOLVER_GPU_KERNEL_H
#define CSOLVER_GPU_KERNEL_H

__global__ void
advct_gpu_kernel_0(float * __restrict__ curv, 
				   float *advx,
				   float *xflux, 
				   float *yflux,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy,
				   int kb, int jm, int im);

__global__ void
advct_gpu_kernel_1(float * __restrict__ xflux, 
				   float * __restrict__ yflux,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb, 
				   const float * __restrict__ aam, 
				   const float * __restrict__ dt, 
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy, 
				   int kb, int jm, int im);

__global__ void
advct_gpu_kernel_2(float * __restrict__ advx, 
				   const float * __restrict__ curv, 
				   const float * __restrict__ xflux, 
				   const float * __restrict__ yflux,
				   const float * __restrict__ v, 
				   const float * __restrict__ dt,
				   const float * __restrict__ aru, 
				   int n_west,
				   int kb, int jm, int im);

__global__ void
advct_gpu_kernel_3(float *advy, 
				   float * __restrict__ xflux, 
				   float * __restrict__ yflux,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb, 
				   const float * __restrict__ aam, 
				   const float * __restrict__ dt, 
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy, 
				   int kb, int jm, int im);

__global__ void
advct_gpu_kernel_4(float * __restrict__ advy, 
				   const float * __restrict__ curv, 
				   const float * __restrict__ xflux, 
				   const float * __restrict__ yflux,
				   const float * __restrict__ u, 
				   const float * __restrict__ dt, 
				   const float * __restrict__ arv, 
				   int n_south, 
				   int kb, int jm, int im);

__global__ void
baropg_gpu_kernel_0(float * __restrict__ rho, 
					const float * __restrict__ rmean,
					int kb, int jm, int im);

__global__ void
baropg_gpu_kernel_1(float * __restrict__ drhox, 
					float * __restrict__ drhoy, 
					const float * __restrict__ dt, 
					const float * __restrict__ rho, 
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const float * __restrict__ dum, 
					const float * __restrict__ dvm, 
					const float * __restrict__ zz,
					float grav, float ramp, 
					int kb, int jm, int im);

__global__ void
baropg_gpu_kernel_2(float * __restrict__ rho, 
					const float * __restrict__ rmean,
					int kb, int jm, int im);


__global__ void 
advave_gpu_kernel_0(//float * __restrict__ advua, 
					const float * __restrict__ d, 
					const float * __restrict__ ua, 
					const float * __restrict__ va,  
					float * __restrict__ fluxua_advua, 
					float * __restrict__ fluxva_advua, 
					float * __restrict__ fluxua_advva, 
					float * __restrict__ fluxva_advva, 
					const float * __restrict__ uab,
					const float * __restrict__ vab, 
					const float * __restrict__ aam2d, 
					//float * __restrict__ tps,  
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const int jm, const int im);

__global__ void 
advave_gpu_kernel_1(float * __restrict__ advua, 
			        const float * __restrict__ fluxua, 
					const float * __restrict__ fluxva, 
				    int jm, int im);

__global__ void 
advave_gpu_kernel_2(float * __restrict__ advva, 
			        float * __restrict__ fluxua, 
					float * __restrict__ fluxva, 
					const float * __restrict__ ua, 
					const float * __restrict__ va, 
					const float * __restrict__ d, 
					const float * __restrict__ aam2d, 
					const float * __restrict__ vab, 
					const float * __restrict__ tps,  
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
				    int jm, int im);

__global__ void 
advave_gpu_kernel_3(float * __restrict__ advva,
			        const float * __restrict__ fluxua, 
					const float * __restrict__ fluxva, 
					//float *uab, float *vab,  
					//float *ua, float *va, 
				    //float *wubot, float *wvbot, 
					//float *curv2d, 
					//float *cbc, float *dx, float *dy, 
					//int mode, 
					int jm, int im);

__global__ void 
advave_gpu_kernel_4(float * __restrict__ wubot, 
					float * __restrict__ wvbot, 
					float * __restrict__ curv2d, 
					const float * __restrict__ uab, 
					const float * __restrict__ vab,  
					const float * __restrict__ cbc,
					const float * __restrict__ ua, 
					const float * __restrict__ va, 
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					int jm, int im);

__global__ void 
advave_gpu_kernel_5(float * __restrict__ advua, 
					float * __restrict__ advva,
					const float * __restrict__ d, 
					const float * __restrict__ ua, 
					const float * __restrict__ va,
					float * __restrict__ aam2d, 
					const float * __restrict__ curv2d,
					const float * __restrict__ aamfrz,  
					const float * __restrict__ aru, 
					const float * __restrict__ arv,  
					const float * __restrict__ dx, 
					const float * __restrict__ dy,  
					float horcon, float aam_init,
					int n_west, int n_south,  
					int jm, int im);


__global__ void
vertvl_gpu_kernel_0(float * __restrict__ xflux, 
					float * __restrict__ yflux, 
				    const float * __restrict__ u, 
					const float * __restrict__ v, 
					const float * __restrict__ dt,
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
				    int kb, int jm, int im);

__global__ void
vertvl_gpu_kernel_1(float * __restrict__ w, 
					const float * __restrict__ vfluxb, 
					const float * __restrict__ vfluxf,
		            const float * __restrict__ xflux, 
					const float * __restrict__ yflux, 
					const float * __restrict__ etf, 
					const float * __restrict__ etb, 
				    const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const float * __restrict__ dz,
				    float dti2, int kb, int jm, int im);

__global__ void
advq_gpu_kernel_0(float * __restrict__ xflux, 
				  float * __restrict__ yflux, 
				  const float * __restrict__ q, 
				  const float * __restrict__ qb,
				  const float * __restrict__ u, 
				  const float * __restrict__ v,
				  const float * __restrict__ aam, 
				  const float * __restrict__ dt, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ h, 
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  int kb, int jm, int im);

__global__ void
advq_gpu_kernel_1(float * __restrict__ qf, 
				  const float * __restrict__ qb, 
				  const float * __restrict__ q,
				  const float * __restrict__ w, 
				  const float * __restrict__ xflux, 
				  const float * __restrict__ yflux,
				  const float * __restrict__ etb, 
				  const float * __restrict__ etf, 
				  const float * __restrict__ art, 
				  const float * __restrict__ dz, 
				  const float * __restrict__ h, 
				  float dti2, int kb, int jm, int im);

__global__ void
profq_gpu_kernel_0(float * __restrict__ utau2, 
				   float * __restrict__ uf, 
				   const float * __restrict__ wusurf, 
				   const float * __restrict__ wvsurf, 
				   const float * __restrict__ wubot, 
				   const float * __restrict__ wvbot,
				   int kb, int jm, int im);

/*
__global__ void
profq_gpu_kernel_1(float *t, float *s,
				   float *u, float *v,
				   float *uf, float *vf,
				   float *q2b, float *q2lb,
				   float *q2,
				   float *rho, float *l,
				   float *prod, float *dh, 
				   float *dtef,
				   float *kq, float *km, float *kh,
				   float *cc, float *boygr,
				   float *ee, float *gg, 
				   float *gh, float *l0, 
				   float *a, float *c, 
				   float *stf,
				   float *sh, float *sm,
				   float *zz, float *h, 
				   float *dzz, float *z,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   int kb, int jm, int im);
*/

__global__ void
profq_gpu_kernel_1(const float * __restrict__ t, 
				   const float * __restrict__ s,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   float * __restrict__ uf, 
				   float * __restrict__ vf,
				   float * __restrict__ q2b, 
				   float * __restrict__ q2lb,
				   const float * __restrict__ q2,
				   const float * __restrict__ rho,
				   const float * __restrict__ etf, 
				   const float * __restrict__ cu_utau2,
				   // float *l,
				   //float *prod, float *dh, 
				   //float *dtef,
				   float * __restrict__ kq, 
				   float * __restrict__ km, 
				   float * __restrict__ kh,
				   //float *cc, float *boygr,
				   //float *ee, float *gg, 
				   //float *gh, float *l0, 
				   //float *a, float *c, 
				   //float *stf,
				   //float *sh, float *sm,
				   const float * __restrict__ h, 
				   const float * __restrict__ z, 
				   const float * __restrict__ zz, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   float umol, 
				   int kb, int jm, int im);

__global__ void
profq_gpu_kernel_2(float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ fsm,
				   int n_north, int n_south, 
				   int n_east, int n_west,
				   int kb, int jm, int im);

__global__ void
advt1_gpu_kernel_0(float * __restrict__ f, 
				   float * __restrict__ fb, 
				   const float * __restrict__ fclim, 
				   int kb, int jm, int im);

__global__ void
advt1_gpu_kernel_1(float * __restrict__ xflux, 
				   float * __restrict__ yflux, 
				   float * __restrict__ zflux,
				   const float * __restrict__ f, 
				   const float * __restrict__ fb,
				   const float * __restrict__ u, 
				   const float * __restrict__ v, 
				   const float * __restrict__ w,
				   const float * __restrict__ aam, 
				   const float * __restrict__ dt, 
				   const float * __restrict__ tsurf,
				   const float * __restrict__ h, 
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy, 
				   const float * __restrict__ dum, 
				   const float * __restrict__ dvm, 
				   const float * __restrict__ art,
				   char var,
				   float tprni, int kb, int jm, int im);


__global__ void
advt1_gpu_kernel_2(const float * __restrict__ xflux, 
				   const float * __restrict__ yflux, 
				   const float * __restrict__ zflux,
				   float * __restrict__ ff, 
				   float * __restrict__ fb, 
				   const float * __restrict__ fclim,
				   const float * __restrict__ etb, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ dt, 
				   const float * __restrict__ relax_aid,
				   const float * __restrict__ art, 
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ zz,
				   float dti2, int kb, int jm, int im);

__global__ void
smol_adif_kernel_0(float * __restrict__ ff, 
				   float * __restrict__ fsm, 
				   int kb, int jm, int im);


__global__ void
smol_adif_kernel_1(const float * __restrict__ ff, 
				   const float * __restrict__ dt,
				   float * __restrict__ xmassflux, 
				   float * __restrict__ ymassflux, 
				   float * __restrict__ zwflux,
				   const float * __restrict__ aru, 
				   const float * __restrict__ arv, 
				   const float * __restrict__ dzz, 
				   float sw, float dti2,
				   int kb, int jm, int im);


__global__ void
advt2_gpu_kernel_0(float * __restrict__ xmassflux, 
				   float * __restrict__ ymassflux, 
				   float * __restrict__ zwflux, 
				   const float * __restrict__ u, 
				   const float * __restrict__ v, 
				   const float * __restrict__ w,
				   float * __restrict__ fb, 
				   float * __restrict__ fbmem, 
				   const float * __restrict__ dt,
				   float * __restrict__ eta, 
				   const float * __restrict__ etb, 
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy, 
				   int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_1(float * __restrict__ xflux, 
				   float * __restrict__ yflux, 
				   float * __restrict__ zflux, 
				   const float * __restrict__ xmassflux, 
				   const float * __restrict__ ymassflux, 
				   const float * __restrict__ zwflux,
				   const float * __restrict__ f, 
				   const float * __restrict__ fbmem, 
				   const float * __restrict__ w, 
				   const float * __restrict__ tsurf, 
				   const float * __restrict__ art, 
				   char var,
				   int itera, int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_2(const float * __restrict__ xflux, 
				   const float * __restrict__ yflux, 
				   const float * __restrict__ zflux,
				   float * __restrict__ ff, 
				   const float * __restrict__ fbmem, 
				   const float * __restrict__ eta, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   const float * __restrict__ dz,
				   float dti2, int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_3(float * __restrict__ eta, 
				   const float * __restrict__ etf,
				   const float * __restrict__ ff, 
				   float * __restrict__ fbmem, 
				   int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_4(float * __restrict__ fb, 
				   const float * __restrict__ fclim,
				   int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_5(const float * __restrict__ fb, 
				   const float * __restrict__ ff, 
				   float *xmassflux, float *ymassflux,
				   float * __restrict__ xflux, 
				   float * __restrict__ yflux,
				   const float * __restrict__ aam, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ h, 
				   const float * __restrict__ art,
				   const float * __restrict__ dum, 
				   const float * __restrict__ dvm, 
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy, 
				   float tprni, float dti2,
				   int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_5(float * __restrict__ ff, 
				   const float * __restrict__ etf,
				   const float * __restrict__ xflux, 
				   const float * __restrict__ yflux,
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   float dti2,
				   int kb, int jm, int im);

__global__ void
advt2_gpu_kernel_6(float * __restrict__ ff, 
				   float * __restrict__ fb, 
				   const float * __restrict__ fclim,
				   const float * __restrict__ etf,
				   const float * __restrict__ xflux, 
				   const float * __restrict__ yflux,
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   float dti2,
				   int kb, int jm, int im);


__global__ void
proft_gpu_kernel_0(float * __restrict__ kh, 
				   float * __restrict__ etf, 
				   float * __restrict__ swrad,
				   float * __restrict__ wfsurf, 
				   float * __restrict__ f, 
				   const float * __restrict__ fsurf,
				   //float *a, float *c, 
				   //float *ee, float *gg,
				   //float *dh, float *rad, 
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz, 
				   const float * __restrict__ z,
				   float dti2, float umol, int ntp, int nbc,
				   int kb, int jm, int im);

__global__ void
dens_gpu_kernel_0(const float * __restrict__ ti, 
				  const float * __restrict__ si, 
				  float * __restrict__ rhoo,
				  const float * __restrict__ zz, 
				  const float * __restrict__ h, 
				  const float * __restrict__ fsm,
				  float tbias, float sbias,
				  float grav, float rhoref,
				  int kb, int jm, int im);

__global__ void
advu_gpu_kernel_0(float * __restrict__ uf, 
				  const float * __restrict__ ub, 
				  const float * __restrict__ u, 
				  const float * __restrict__ v, 
				  const float * __restrict__ w,
				  const float * __restrict__ advx, 
				  const float * __restrict__ egf, 
				  const float * __restrict__ egb, 
				  const float * __restrict__ etf, 
				  const float * __restrict__ etb,
				  const float * __restrict__ dt, 
				  const float * __restrict__ e_atmos, 
				  const float * __restrict__ drhox,
				  const float * __restrict__ h, 
				  const float * __restrict__ cor, 
				  const float * __restrict__ aru, 
				  const float * __restrict__ dy, 
				  const float * __restrict__ dz, 
				  float grav, float dti2, 
				  int kb, int jm, int im);

__global__ void
advv_gpu_kernel_0(float * __restrict__ vf, 
				  const float * __restrict__ v, 
				  const float * __restrict__ vb,
				  const float * __restrict__ u, 
				  const float * __restrict__ w,
				  const float * __restrict__ etb, 
				  const float * __restrict__ etf,
				  const float * __restrict__ egf, 
				  const float * __restrict__ egb, 	
				  const float * __restrict__ advy,
				  const float * __restrict__ dt,   
				  const float * __restrict__ e_atmos, 
				  const float * __restrict__ drhoy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ arv, 
				  const float * __restrict__ dx, 
				  const float * __restrict__ dz, 
				  const float * __restrict__ h, 
				  float grav, float dti2,
				  int kb, int jm, int im);

__global__ void
profu_gpu_kernel_0(
				   //float *dh, float *tps,
				   //float *a, float *c, 
				   //float *ee, float *gg, 
				   float * __restrict__ uf, 
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb,
				   const float * __restrict__ km, 
				   const float * __restrict__ etf,
				   const float * __restrict__ wusurf, 
				   float * __restrict__ wubot, 
				   const float * __restrict__ cbc, 
				   const float * __restrict__ dum,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float dti2, float umol,
				   int kb, int jm, int im);

__global__ void
profv_gpu_kernel_0(
				   //float *dh, float *tps, 
				   //float *a, float *c, 
				   //float *ee, float *gg,
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb, 
				   float * __restrict__ vf, 
				   const float * __restrict__ km, 
				   const float * __restrict__ etf,
				   const float * __restrict__ wvsurf, 
				   float * __restrict__ wvbot, 
				   const float * __restrict__ cbc, 
				   const float * __restrict__ dvm,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float dti2, float umol,
				   int kb, int jm, int im);

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#endif

