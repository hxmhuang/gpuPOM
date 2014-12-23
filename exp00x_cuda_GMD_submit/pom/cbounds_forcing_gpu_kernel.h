#ifndef CBOUNDS_FORCING_GPU_KERNEL_H
#define CBOUNDS_FORCING_GPU_KERNEL_H

__global__ void
bcond_gpu_kernel_1(float * __restrict__ elf, 
				   const float * __restrict__ fsm,
				   int jm, int im);

__global__ void
bcond_gpu_kernel_2_0(float *uaf, float *vaf, float *el,
				   float *uabe, float *uabw, 
				   float *vabs, float *vabn, 
				   float *ele, float *elw,
				   float *els, float *eln,
				   float *h, 
				   float rfe, float rfw, 
				   float rfs, float rfn,
				   float ramp, float grav,  
				   int n_east, int n_west,
				   int n_south, int n_north,
				   int jm, int im);

__global__ void
bcond_gpu_kernel_2_1(float *uaf, float *dum,
					 int n_north, int n_south,
				     int jm, int im);

__global__ void
bcond_gpu_kernel_2_2(float *vaf, float *dvm,
					 int n_east, int n_west,
				     int jm, int im);

__global__ void
bcond_gpu_kernel_2_3(float *uaf, float *vaf,
				     float *dum, float *dvm,
				     int jm, int im);

__global__ void
bcond_gpu_kernel_3(float *uf, float *vf,
				   float *dum, float *dvm,
				   int kb, int jm, int im);


__global__ void
bcond_gpu_kernel_4_0(float *u, float *uf,
				     float *v, float *vf, 
				     float *t, float *s, 
				     float *w, float *dt, 
					 float *tobe, float *sobe,
					 float *tobw, float *sobw,
					 float *tobs, float *sobs,
					 float *tobn, float *sobn,
				     float *tbe, float *sbe, 
				     float *tbw, float *sbw,
				     float *tbs, float *sbs,
				     float *tbn, float *sbn,
				     float *dx, float *dy,
				     float *zz, float *frz,
				     float dti, 
				     int n_east, int n_west, 
				     int n_south, int n_north,
				     int kb, int jm, int im);

__global__ void
bcond_gpu_kernel_4_1(float *uf, float *vf, 
					 float *fsm, 
					 int kb, int jm, int im);


__global__ void
bcond_gpu_kernel_5(float *w, float *fsm,
				   int kb, int jm, int im);




__global__ void
bcond_gpu_kernel_6_0(float *u, float *v, 
				     float *uf, float *vf,
				     float *q2, float *q2l,
				     float *dx, float *dy, 
				     float dti, float small,
				     int n_east, int n_west, 
				     int n_south, int n_north,
				     int kb, int jm, int im);

__global__ void
bcond_gpu_kernel_6_1(float *uf, float *vf,
				     float *fsm,
				     int kb, int jm, int im);
#endif
