#include<math.h>
#include"cbounds_forcing_gpu.h"
#include"cbounds_forcing_gpu_kernel.h"

#include"cu_data.h"
#include"cparallel_mpi_gpu.h"

#include"utils.h"
#include"data.h"
#include"timer_all.h"

// bounds_forcing.f
//

// spcify variable boundary conditions, atmospheric forcing, restoring

/*
!_______________________________________________________________________
      subroutine bcond(idx)
! apply open boundary conditions
! closed boundary conditions are automatically enabled through
! specification of the masks, dum, dvm and fsm, in which case the open
! boundary conditions, included below, will be overwritten
      implicit none
      include 'pom.h'
      integer idx
      integer i,j,k
      real ga,u1,wm
      real hmax

*/

__global__ void
bcond_gpu_kernel_1(float * __restrict__ elf, 
				   const float * __restrict__ fsm,
				   int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			elf[j][i] *= fsm[j][i];	
		}
	}
	*/

	if (j < jm && i < im){
		elf[j_off+i] *= fsm[j_off+i];	
	}

	return;

}

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
				   int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	/*
	if (n_east == -1){
		for (j = 1; j < jmm1; j++){
			uaf[j][im-1] = uabe[j]+rfe
								  *sqrtf(grav/h[j][imm1-1])
								  *(el[j][imm1-1]-ele[j]);
			uaf[j][im-1] = ramp*uaf[j][im-1];
			vaf[j][im-1] = 0;

		}
	}
	if (n_west == -1){
		for (j = 1; j < jmm1; j++){
			uaf[j][1] = uabw[j]-rfw*sqrtf(grav/h[j][1])
								   *(el[j][1]-elw[j]);		
			uaf[j][1] = ramp*uaf[j][1];
			uaf[j][0] = uaf[j][1];
			vaf[j][0] = 0;
		}
	}
	*/

	if (n_east == -1){
		if (j > 0 && j < jmm1 && i == (im-1)){
			uaf[j_off+(im-1)] = uabe[j]
							   +rfe*sqrtf(grav/h[j_off+(imm1-1)])
								   *(el[j_off+(imm1-1)]-ele[j]);
			uaf[j_off+(im-1)] *= ramp;
			vaf[j_off+(im-1)] = 0;
		}
	}

	if (n_west == -1){
		if (j > 0 && j < jmm1 && i == 0){
			uaf[j_off+1] = uabw[j]
						  -rfw*sqrtf(grav/h[j_off+1])
							  *(el[j_off+1]-elw[j]);
			uaf[j_off+1] *= ramp;
			uaf[j_off] = uaf[j_off+1];
			vaf[j_off] = 0;
		}
	}


	/*
	if (n_north == -1){
		for (i = 1; i < imm1; i++){
			vaf[jm-1][i] = vabn[i]+rfn
								  *sqrtf(grav/h[jmm1-1][i])	
								  *(el[jmm1-1][i]-eln[i]);
			vaf[jm-1][i] = ramp*vaf[jm-1][i];
			uaf[jm-1][i] = 0;
		}
	}

	if (n_south == -1){
		for (i = 1; i < imm1; i++){
			vaf[1][i] = vabs[i]-rfs*sqrtf(grav/h[1][i])
								   *(el[1][i]-els[i]);
			vaf[1][i] = ramp*vaf[1][i];
			vaf[0][i] = vaf[1][i];
			uaf[0][i] = 0;
		}
	}
	*/

	//__syncthreads();
	
	if (n_north == -1){
		if (i > 0 && i < imm1 && j == (jm-1)){
			vaf[jm_1_off+i] = vabn[i]
							 +rfn*sqrtf(grav/h[jmm1_1_off+i])
								 *(el[jmm1_1_off+i]-eln[i]);
			vaf[jm_1_off+i] *= ramp;
			uaf[jm_1_off+i] = 0;
		}
	}
	
	if (n_south == -1){
		if (i > 0 && i < imm1 && j == 0){
			vaf[im+i] = vabs[i]
					   -rfs*sqrtf(grav/h[im+i])
						   *(el[im+i]-els[i]);
			vaf[im+i] *= ramp;
			vaf[i] = vaf[im+i];
			uaf[i] = 0;
		}
	}

	
	
	return;
}

__global__ void
bcond_gpu_kernel_2_1(float *uaf, float *dum,
					 int n_north, int n_south,
				     int jm, int im){
	
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int jmm1 = jm-1; 

	if (n_north == -1){
		if (i < im && j == jm-1){
			uaf[jm_1_off+i] = uaf[jmm1_1_off+i];
			dum[jm_1_off+i] = 1.f;
		}
	}
	if (n_south == -1){
		if (i < im && j == 0){
			uaf[i] = uaf[im+i];
			dum[i] = 1.f;
		}
	}
}

__global__ void
bcond_gpu_kernel_2_2(float *vaf, float *dvm,
					 int n_east, int n_west,
				     int jm, int im){
	
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int imm1 = im-1; 

	if (n_east == -1){
		if (j < jm && i == im-1){
			vaf[j_off+im-1] = vaf[j_off+imm1-1];
			dvm[j_off+im-1] = 1.f;
		}
	}
	if (n_west == -1){
		if (j < jm && i == 0){
			vaf[j_off] = vaf[j_off+1];
			dvm[j_off] = 1.f;
		}
	}
}

__global__ void
bcond_gpu_kernel_2_3(float *uaf, float *vaf,
				     float *dum, float *dvm,
				     int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	
	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			uaf[j][i] *= dum[j][i];	
			vaf[j][i] *= dvm[j][i];
		}
	}
	*/

	if (j < jm && i < im){
		uaf[j_off+i] *= dum[j_off+i];
		vaf[j_off+i] *= dvm[j_off+i];
	}
}

__global__ void
bcond_gpu_kernel_3_0(float *uf, float *vf,
					 float *u, float *v,
					 float *h,
					 int n_east, int n_west,
					 int n_south, int n_north,
					 int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	int imm1 = im-1;
	int jmm1 = jm-1;
	int kbm1 = kb-1;

	float ga;
	const float hmax = 8000.f;

	/*
	if (n_east == -1){
		for (k = 0; k < kbm1; k++){
			for (j = 1; j < jmm1; j++){
				ga = sqrtf(h[j][im-1]/hmax);
				uf[k][j][im-1] = ga*(0.25f*u[k][j-1][imm1-1]
									+0.5f*u[k][j][imm1-1]
									+0.25f*u[k][j+1][imm1-1])
								+(1.f-ga)*(0.25f*u[k][j-1][im-1]
										  +0.5f*u[k][j][im-1]
										  +0.25f*u[k][j+1][im-1]);
				vf[k][j][im-1] = 0;
			}
		}
	}
	*/

	if (n_east == -1){
		for (k = 0; k < kbm1; k++){
			if (j > 0 && j < jm && i == im-1){
				ga = sqrtf(h[j_off+im-1]/hmax);
				uf[k_off+j_off+im-1] = ga*(0.25f*u[k_off+j_1_off+imm1-1]
										  +0.5f*u[k_off+j_off+imm1-1]
										  +0.25f*u[k_off+j_A1_off+imm1-1])
									  +(1.f-ga)
										*(0.25f*u[k_off+j_1_off+im-1]
										 +0.5f*u[k_off+j_off+im-1]
										 +0.25f*u[k_off+j_A1_off+im-1]);

				vf[k_off+j_off+im-1] = 0;
			}
		}
	}

	/*
	if (n_west == -1){
		for (k = 0; k < kbm1; k++){
			for (j = 1; j < jmm1; j++){
				ga = sqrtf(h[j][0]/hmax);
				uf[k][j][1] = ga*(0.25f*u[k][j-1][2]
								 +0.5f*u[k][j][2]
								 +0.25f*u[k][j+1][2])
							 +(1.f-ga)*(0.25f*u[k][j-1][1]
									   +0.5f*u[k][j][1]
									   +0.25f*u[k][j+1][1]);
				uf[k][j][0] = uf[k][j][1];
				vf[k][j][0] = 0;
			}
		}
	}
	*/

	if (n_west == -1){
		for (k = 0; k < kbm1; k++){
			if (j > 0 && j < jmm1 && i == 0){
				ga = sqrtf(h[j_off]/hmax);
				uf[k_off+j_off+1] = ga*(0.25f*u[k_off+j_1_off+2]
									   +0.5f*u[k_off+j_off+2]
									   +0.25f*u[k_off+j_A1_off+2])
								   +(1.f-ga)
									  *(0.25f*u[k_off+j_1_off+1]
									   +0.5f*u[k_off+j_off+1]
									   +0.25f*u[k_off+j_A1_off+1]);

				uf[k_off+j_off] = uf[k_off+j_off+1];
				vf[k_off+j_off] = 0;
			}
		}
	}

	/*
	if (n_north == -1){
		for (k = 0; k < kbm1; k++){
			for (i = 1; i < imm1; i++){
				ga = sqrtf(h[jm-1][i]/hmax);
				vf[k][jm-1][i] = ga*(0.25f*v[k][jmm1-1][i-1]
									+0.5f*v[k][jmm1-1][i]
									+0.25f*v[k][jmm1-1][i+1])
								+(1.f-ga)*(0.25f*v[k][jm-1][i-1]
										  +0.5f*v[k][jm-1][i]
										  +0.25f*v[k][jm-1][i+1]);
				uf[k][jm-1][i] = 0;
			}
		}
	}
	*/

	if (n_north == -1){
		for (k = 0; k < kbm1; k++){
			if (i > 0 && i < imm1 && j == jm-1){
				ga = sqrtf(h[jm_1_off+i]/hmax);	
				vf[k_off+jm_1_off+i] = ga*(0.25f*v[k_off+jmm1_1_off+i-1]
										  +0.5f*v[k_off+jmm1_1_off+i]
										  +0.25f*v[k_off+jmm1_1_off+i+1])
									  +(1.f-ga)
										 *(0.25f*v[k_off+jm_1_off+i-1]
										  +0.5f*v[k_off+jm_1_off+i]
										  +0.25f*v[k_off+jm_1_off+i+1]);
				uf[k_off+jm_1_off+i] = 0;
			}
		}
	}

	/*
	if (n_south == -1){
		for (k = 0; k < kbm1; k++){
			for (i = 1; i < imm1; i++){
				ga = sqrtf(h[0][i]/hmax);	
				vf[k][1][i] = ga*(0.25f*v[k][2][i-1]
								 +0.5f*v[k][2][i]
								 +0.25*v[k][2][i+1])
							 +(1.f-ga)*(0.25f*v[k][1][i-1]
									   +0.5f*v[k][1][i]
									   +0.25f*v[k][1][i+1]);
				vf[k][0][i] = vf[k][1][i];
				uf[k][0][i] = 0;
			}
		}
	}
	*/

	if (n_south == -1){
		for (k = 0; k < kbm1; k++){
			if (i > 0 && i < imm1 && j == 0){
				ga = sqrtf(h[i]/hmax);
				vf[k_off+im+i] = ga*(0.25f*v[k_off+2*im+i-1]
									+0.5f*v[k_off+2*im+i]
									+0.25f*v[k_off+2*im+i+1])
								+(1.f-ga)
								   *(0.25f*v[k_off+im+i-1]
									+0.5f*v[k_off+im+i]
									+0.25f*v[k_off+im+i+1]);
				vf[k_off+i] = vf[k_off+im+i];
				uf[k_off+i] = 0;
			}
		}
	}

}

__global__ void
bcond_gpu_kernel_3_1(float *uf, float *vf,
				     float *dum, float *dvm,
				     int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	int kbm1 = kb-1;

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = uf[k][j][i]*dum[j][i];	
				vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			uf[k_off+j_off+i] *= dum[j_off+i];
			vf[k_off+j_off+i] *= dvm[j_off+i];
		}
	}


}


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
				     int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	
	int kbm1 = kb-1;
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	float u1, wm;
	/*
   
	 */

	if (n_west == -1){
		for (k = 0; k < kbm1; k++){
			if (j < jm && i == 0){
				u1 = 2.0f*u[k_off+j_off+1]*dti
						 /(dx[j_off]+dx[j_off+1]);
				if (u1 >= 0.0f){
					uf[k_off+j_off] = t[k_off+j_off]
									 -u1*(t[k_off+j_off]
										 -tbw[k*jm+j]);	

					vf[k_off+j_off] = s[k_off+j_off]
									 -u1*(s[k_off+j_off]
										 -sbw[k*jm+j]);

				}else{
					uf[k_off+j_off] = t[k_off+j_off]
									 -u1*(t[k_off+j_off+1]
										 -t[k_off+j_off]);	

					vf[k_off+j_off] = s[k_off+j_off]
									 -u1*(s[k_off+j_off+1]
										 -s[k_off+j_off]);

					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+j_off+1]
								  +w[k_A1_off+j_off+1])
								 *dti
								 /((zz[k-1]-zz[k+1])
								   *dt[j_off+1]);

						uf[k_off+j_off] = uf[k_off+j_off]
										 -wm*(t[k_1_off+j_off+1]
											 -t[k_A1_off+j_off+1]);

						vf[k_off+j_off] = vf[k_off+j_off]
										  -wm*(s[k_1_off+j_off+1]
											  -s[k_A1_off+j_off+1]);
					}
				}
			}
		}
		if (nfw > 3){
			for (k = 0; k < kbm1; k++){
				if (j < jm && i < nfw){
					uf[k_off+j_off+i] = uf[k_off+j_off+i]
											*(1.f-frz[j_off+i])
									   +(tobw[k_off+j_off+i]*frz[j_off+i]);

					vf[k_off+j_off+i] = vf[k_off+j_off+i]
											*(1.f-frz[j_off+i])
									   +(sobw[k_off+j_off+i]*frz[j_off+i]);
				}
			}
		}
	}

	if (n_east == -1){
		for (k = 0; k < kbm1; k++){
			if (j < jm && i == (im-1)){
				u1 = 2.0f*u[k_off+j_off+(im-1)]
						 *dti/(dx[j_off+(im-1)]
							  +dx[j_off+(imm1-1)]);
				if (u1 <= 0.0f){
					uf[k_off+j_off+(im-1)] = t[k_off+j_off+(im-1)]
											-u1*(tbe[k*jm+j]
												-t[k_off+j_off+(im-1)]);	

					vf[k_off+j_off+(im-1)] = s[k_off+j_off+(im-1)]
											-u1*(sbe[k*jm+j]
												-s[k_off+j_off+(im-1)]);
				}else{
					uf[k_off+j_off+(im-1)] = t[k_off+j_off+(im-1)]
											-u1*(t[k_off+j_off+(im-1)]
												-t[k_off+j_off+(imm1-1)]);	

					vf[k_off+j_off+(im-1)] = s[k_off+j_off+(im-1)]
											-u1*(s[k_off+j_off+(im-1)]
											    -s[k_off+j_off+(imm1-1)]);

					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+j_off+(imm1-1)]
								  +w[k_A1_off+j_off+(imm1-1)])
								 *dti
								 /((zz[k-1]-zz[k+1])
								  *dt[j_off+(imm1-1)]);

						uf[k_off+j_off+(im-1)] = 
								uf[k_off+j_off+(im-1)]
							   -wm*(t[k_1_off+j_off+(imm1-1)]
								   -t[k_A1_off+j_off+(imm1-1)]);

						vf[k_off+j_off+(im-1)] = 
								vf[k_off+j_off+(im-1)]
							   -wm*(s[k_1_off+j_off+(imm1-1)]
								   -s[k_A1_off+j_off+(imm1-1)]);
					}
				}
			}
		}
		if (nfe > 3){
			for (k = 0; k < kbm1; k++){
				if (j < jm && i >= im-nfe && i < im){
					int ii = im-i-1;	
					uf[k_off+j_off+i] = uf[k_off+j_off+i]
											*(1.f-frz[j_off+i])
									   +(tobe[k_off+j_off+ii]
											*frz[j_off+i]);

					vf[k_off+j_off+i] = vf[k_off+j_off+i]
											*(1.f-frz[j_off+i])
										+(sobe[k_off+j_off+ii]
											*frz[j_off+i]);
				}
			}
		}
	}

	if (n_north == -1){
		for (k = 0; k < kbm1; k++){
			if (i < im && j == (jm-1)){
				u1 = 2.0f*v[k_off+jm_1_off+i]
						 *dti
						 /(dy[jm_1_off+i]
						  +dy[jmm1_1_off+i]);	

				if (u1 <= 0){
					uf[k_off+jm_1_off+i] = t[k_off+jm_1_off+i]
										  -u1*(tbn[k*im+i]
											  -t[k_off+jm_1_off+i]);

					vf[k_off+jm_1_off+i] = s[k_off+jm_1_off+i]
										  -u1*(sbn[k*im+i]
											  -s[k_off+jm_1_off+i]);
				}else{
					uf[k_off+jm_1_off+i] = t[k_off+jm_1_off+i]
										  -u1*(t[k_off+jm_1_off+i]
											  -t[k_off+jmm1_1_off+i]);

					vf[k_off+jm_1_off+i] = s[k_off+jm_1_off+i]
										  -u1*(s[k_off+jm_1_off+i]
											  -s[k_off+jmm1_1_off+i]);

					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+jmm1_1_off+i]
								  +w[k_A1_off+jmm1_1_off+i])
								 *dti
								 /((zz[k-1]-zz[k+1])
									*dt[jmm1_1_off+i]);	

						uf[k_off+jm_1_off+i] = uf[k_off+jm_1_off+i]
											  -wm*(t[k_1_off+jmm1_1_off+i]
												  -t[k_A1_off+jmm1_1_off+i]);

						vf[k_off+jm_1_off+i] = vf[k_off+jm_1_off+i]
											  -wm*(s[k_1_off+jmm1_1_off+i]
												  -s[k_A1_off+jmm1_1_off+i]);
					}
				}
			}
		}

		if (nfn > 3){
			for (k = 0; k < kbm1; k++){
				if (i < im && j >= jm-nfn && j < jm){
					int jj = jm-j-1;	
					uf[k_off+j_off+i] = uf[k_off+j_off+i]
											*(1.f-frz[j_off+i])
									   +(tobn[k_off+jj*im+i]
											*frz[j_off+i]);

					vf[k_off+j_off+i] = vf[k_off+j_off+i]
											*(1.f-frz[j_off+i])
									   +(sobn[k_off+jj*im+i]
											*frz[j_off+i]);
				}
			}
		}
	}


	if (n_south == -1){
		for (k = 0; k < kbm1; k++){
			if (i < im && j == 0){
				u1=2.0f*v[k_off+1*im+i]*dti/(dy[i]+dy[1*im+i]);	
				if (u1 >= 0.0f){
					uf[k_off+i] = t[k_off+i]-u1*(t[k_off+i]-tbs[k*im+i]);	
					vf[k_off+i] = s[k_off+i]-u1*(s[k_off+i]-sbs[k*im+i]);
				}else{
					uf[k_off+i] = t[k_off+i]-u1*(t[k_off+1*im+i]-t[k_off+i]);	
					vf[k_off+i] = s[k_off+i]-u1*(s[k_off+1*im+i]-s[k_off+i]);
					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+1*im+i]+w[k_A1_off+1*im+i])
								 *dti
								 /((zz[k-1]-zz[k+1])*dt[1*im+i]);

						uf[k_off+i] = uf[k_off+i]
									 -wm*(t[k_1_off+1*im+i]
										 -t[k_A1_off+1*im+i]);

						vf[k_off+i] = vf[k_off+i]
									 -wm*(s[k_1_off+1*im+i]
										 -s[k_A1_off+1*im+i]);
					}
				}
			}
		}

		if (nfs > 3){
			for (k = 0; k < kbm1; k++){
				if (i < im && j < nfs){
					uf[k_off+j_off+i] = (uf[k_off+j_off+i]
											*(1.f-frz[j_off+i]))
										+(tobs[k_off+j_off+i]
											*frz[j_off+i]);

					vf[k_off+j_off+i] = (vf[k_off+j_off+i]
											*(1.f-frz[j_off+i]))
										+(sobs[k_off+j_off+i]
											*frz[j_off+i]);
				}
			}
		}
	}


}

__global__ void
bcond_gpu_kernel_4_1(float *uf, float *vf, 
					 float *fsm, 
					 int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	int kbm1 = kb-1;

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] *= fsm[j][i];
				vf[k][j][i] *= fsm[j][i];
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			uf[k_off+j_off+i] *= fsm[j_off+i];
			vf[k_off+j_off+i] *= fsm[j_off+i];
		}
	}

}

__global__ void
bcond_gpu_kernel_5(float *w, float *fsm,
				   int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	int kbm1 = kb-1;
	//int jmm1 = jm-1; 
	//int imm1 = im-1; 
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				w[k][j][i] *= fsm[j][i];	
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			w[k_off+j_off+i] *= fsm[j_off+i];	
		}
	}
}

__global__ void
bcond_gpu_kernel_6_0(float *u, float *v, 
				     float *uf, float *vf,
				     float *q2, float *q2l,
				     float *dx, float *dy, 
				     float dti, float small,
				     int n_east, int n_west, 
				     int n_south, int n_north,
				     int kb, int jm, int im){
	float u1;
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	//int kbm1 = kb-1;
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	//modify uf vf
	/*
	if (n_east == -1){
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				u1 = 2.0f*u[k][j][im-1]*dti
					/(dx[j][im-1]+dx[j][imm1-1]);	
				if (u1 <= 0.0f){
					uf[k][j][im-1] = q2[k][j][im-1]-u1*(small-q2[k][j][im-1]);
					vf[k][j][im-1] = q2l[k][j][im-1]-u1*(small-q2l[k][j][im-1]);
				}else{
					uf[k][j][im-1] = q2[k][j][im-1]
						         -u1*(q2[k][j][im-1]-q2[k][j][imm1-1]);	

					vf[k][j][im-1] = q2l[k][j][im-1]
						         -u1*(q2l[k][j][im-1]-q2l[k][j][imm1-1]);
				}
			}
		}
	}
	*/

	if (n_east == -1){
		for (k = 0; k < kb; k++){
			if (j < jm && i == (im-1)){
				u1 = 2.0f*u[k_off+j_off+(im-1)]*dti
						 /(dx[j_off+(im-1)]+dx[j_off+(imm1-1)]);
				if (u1 <= 0){
					uf[k_off+j_off+(im-1)] = q2[k_off+j_off+(im-1)]
											-u1*(small
												-q2[k_off+j_off+(im-1)]);

					vf[k_off+j_off+(im-1)] = q2l[k_off+j_off+(im-1)]
											-u1*(small
												-q2l[k_off+j_off+(im-1)]);
				}else{
					uf[k_off+j_off+(im-1)] = q2[k_off+j_off+(im-1)]
											-u1*(q2[k_off+j_off+(im-1)]
												-q2[k_off+j_off+(imm1-1)]);
					vf[k_off+j_off+(im-1)] = q2l[k_off+j_off+(im-1)]
											-u1*(q2l[k_off+j_off+(im-1)]
												-q2l[k_off+j_off+(imm1-1)]);
				}
			}
		}
	}

	/*
	if (n_west == -1){
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				u1 = 2.0f*u[k][j][1]*dti/(dx[j][0]+dx[j][1]);	
				if (u1 >= 0.0f){
					uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][0]-small);
					vf[k][j][0] = q2l[k][j][0]-u1*(q2l[k][j][0]-small);
				}else{
					uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][1]-q2[k][j][0]);	
					vf[k][j][0] = q2l[k][j][0] - u1*(q2l[k][j][1]-q2l[k][j][0]);
				}
			}
		}
	}
	*/

	if (n_west == -1){
		for (k = 0; k < kb; k++){
			if (j < jm && i == 0){
				u1 = 2.0f*u[k_off+j_off+1]
						 *dti/(dx[j_off]+dx[j_off+1]);
				if (u1 >= 0){
					uf[k_off+j_off] = q2[k_off+j_off]
									 -u1*(q2[k_off+j_off]-small);	
					vf[k_off+j_off] = q2l[k_off+j_off]
									 -u1*(q2l[k_off+j_off]-small);
				}else{
					uf[k_off+j_off] = q2[k_off+j_off]
									 -u1*(q2[k_off+j_off+1]
										 -q2[k_off+j_off]);	

					vf[k_off+j_off] = q2l[k_off+j_off]
									 -u1*(q2l[k_off+j_off+1]
										 -q2l[k_off+j_off]);
				}
			}
		}
	}

	/*
	if (n_south == -1){
		for (k = 0; k < kb; k++){
			for (i = 0; i < im; i++){
				u1 = 2.0f*v[k][1][i]*dti/(dy[0][i]+dy[1][i]);	
				if (u1 >= 0.0f){
					uf[k][0][i] = q2[k][0][i]-u1*(q2[k][0][i]-small);	
					vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][0][i]-small);
				}else{
					uf[k][0][i] = q2[k][0][i]-u1*(q2[k][1][i]-q2[k][0][i]);	
					vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][1][i]-q2l[k][0][i]);
				}
			}
		}
	}
	*/

	if (n_south == -1){
		for (k = 0; k < kb; k++){
			if (i < im && j == 0){
				u1 = 2.0f*v[k_off+1*im+i]*dti/(dy[i]+dy[im+i]);
				if (u1 >= 0){
					uf[k_off+i] = q2[k_off+i]
								 -u1*(q2[k_off+i]-small);	
					vf[k_off+i] = q2l[k_off+i]
								 -u1*(q2l[k_off+i]-small);	
				}else{
					uf[k_off+i] = q2[k_off+i]
								 -u1*(q2[k_off+1*im+i]
									 -q2[k_off+i]);	
					vf[k_off+i] = q2l[k_off+i]
								 -u1*(q2l[k_off+1*im+i]
									 -q2l[k_off+i]);
				}
			}
		}
	}

	/*
	if (n_north == -1){
		for (k = 0; k < kb; k++){
			for (i = 0; i < im; i++){
				u1 = 2.0f*v[k][jm-1][i]*dti/(dy[jm-1][i]+dy[jmm1-1][i]);		
				if (u1 <= 0.0f){
					uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(small-q2[k][jm-1][i]);	
					vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(small-q2l[k][jm-1][i]);
				}else{
					uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(q2[k][jm-1][i]-q2[k][jmm1-1][i]);	
					vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(q2l[k][jm-1][i]-q2l[k][jmm1-1][i]);
				}

			}
		}
	}
	*/

	if (n_north == -1){
		for (k = 0; k < kb; k++){
			if (i < im && j == (jm-1)){
				u1 = 2.0f*v[k_off+jm_1_off+i]*dti
						 /(dy[jm_1_off+i]+dy[jmm1_1_off+i]);
				if (u1 <= 0){
					uf[k_off+jm_1_off+i] = q2[k_off+jm_1_off+i]
										  -u1*(small
											  -q2[k_off+jm_1_off+i]);	
					vf[k_off+jm_1_off+i] = q2l[k_off+jm_1_off+i]
										  -u1*(small
											  -q2l[k_off+jm_1_off+i]);
				}else{
					uf[k_off+jm_1_off+i] = q2[k_off+jm_1_off+i]
										  -u1*(q2[k_off+jm_1_off+i]
											  -q2[k_off+jmm1_1_off+i]);

					vf[k_off+jm_1_off+i] = q2l[k_off+jm_1_off+i]
										  -u1*(q2l[k_off+jm_1_off+i]
											  -q2l[k_off+jmm1_1_off+i]);
				}
			}
		}
	}


}


__global__ void
bcond_gpu_kernel_6_1(float *uf, float *vf,
				     float *fsm,
				     int kb, int jm, int im){
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = uf[k][j][i]*fsm[j][i];
				vf[k][j][i] = vf[k][j][i]*fsm[j][i];
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			uf[k_off+j_off+i] = uf[k_off+j_off+i]*fsm[j_off+i];
			vf[k_off+j_off+i] = vf[k_off+j_off+i]*fsm[j_off+i]; 
		}
	}
}


void bcond_gpu(int idx){

#ifndef TIME_DISABLE
	struct timeval start_bcond,
				   end_bcond;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond);
#endif

	/*
    int i,j,k;
    float ga,u1,wm;
    float hmax;
	*/

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
					  (j_size+block_j_2D-1)/block_j_2D);

	/*
      if(idx.eq.1) then

! eExternal (2-D) elevation boundary conditions
        do j=1,jm
          if(n_west.eq.-1) elf(1,j)=elf(2,j)
          if(n_east.eq.-1) elf(im,j)=elf(imm1,j)
        end do

        do i=1,im
          if(n_south.eq.-1) elf(i,1)=elf(i,2)
          if(n_north.eq.-1) elf(i,jm)=elf(i,jmm1)
        end do

        do j=1,jm
          do i=1,im
            elf(i,j)=elf(i,j)*fsm(i,j)
          end do
        end do

        return
	*/
	if (idx == 1){

#ifndef TIME_DISABLE
	struct timeval start_bcond_1,
				   end_bcond_1;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_1);
#endif
		/*
		if (n_west == -1){
			for (j = 0; j < jm; j++){
				elf[j][0] = elf[j][1];	
			}
		}
		if (n_east == -1){
			for (j = 0; j < jm; j++){
				elf[j][im-1] = elf[j][imm1-1];	
			}
		}
		if (n_north == -1){
			for (i = 0; i < im; i++){
				elf[jm-1][i] = elf[jmm1-1][i];	
			}
		}
		if (n_south == -1){
			for (i = 0; i < im; i++){
				elf[0][i] = elf[1][i];	
			}
		}

		if (iperx != 0){
			xperi2d_mpi(elf, im, jm);
		}

		if (ipery != 0){
			yperi2d_mpi(elf, im, jm);	
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				elf[j][i] = elf[j][i]*fsm[j][i];	
			}
		}
		*/
		/*
		checkCudaErrors(cudaMemcpy(d_elf, elf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/
		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2D(d_elf, im*sizeof(float), 
									     d_elf+1, im*sizeof(float),
									     sizeof(float), jm,
									     cudaMemcpyDeviceToDevice));
		}
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2D(d_elf+(im-1), im*sizeof(float),
									     d_elf+(imm1-1), im*sizeof(float),
									     sizeof(float), jm,
									     cudaMemcpyDeviceToDevice));
		}
		if (n_north == -1){
			checkCudaErrors(cudaMemcpy(d_elf+(jm-1)*im, d_elf+(jmm1-1)*im,
									   im*sizeof(float), 
									   cudaMemcpyDeviceToDevice));	
		}
		if (n_south == -1){
			checkCudaErrors(cudaMemcpy(d_elf, d_elf+im,
									   im*sizeof(float), 
									   cudaMemcpyDeviceToDevice));	
		
		}

		if (iperx != 0)
			//xperi2d_mpi_gpu(d_elf, im, jm);
			xperi2d_cuda_aware_mpi(d_elf, im, jm);
		if (ipery != 0)
			yperi2d_mpi_gpu(d_elf, im, jm);

		bcond_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
				d_elf, d_fsm, jm, im);

		/*
		checkCudaErrors(cudaMemcpy(elf, d_elf, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/
		/*
		bcond_gpu_kernel_1(float *d_elf, float *fsm,
				   int jm, int im){
		*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_1);
		bcond_time_1 += time_consumed(&start_bcond_1, 
								      &end_bcond_1);
#endif
		//return;
	}
/*
      else if(idx.eq.2) then

! external (2-D) velocity boundary conditions
        do j=2,jmm1
          ! west
          if(n_west.eq.-1) then
            uaf(2,j)=uabw(j)-rfw*sqrt(grav/h(2,j))*(el(2,j)-elw(j))
            uaf(2,j)=ramp*uaf(2,j)
            uaf(1,j)=uaf(2,j)
            vaf(1,j)=0.e0
          end if

          ! east
          if(n_east.eq.-1) then
            uaf(im,j)=uabe(j)
     $                     +rfe*sqrt(grav/h(imm1,j))*(el(imm1,j)-ele(j))
            uaf(im,j)=ramp*uaf(im,j)
            vaf(im,j)=0.e0
          end if
        end do

        do i=2,imm1
          ! south
          if(n_south.eq.-1) then
            vaf(i,2)=vabs(i)-rfs*sqrt(grav/h(i,2))*(el(i,2)-els(i))
            vaf(i,2)=ramp*vaf(i,2)
            vaf(i,1)=vaf(i,2)
            uaf(i,1)=0.e0
          end if

          ! north
          if(n_north.eq.-1) then
            vaf(i,jm)=vabn(i)
     $                     +rfn*sqrt(grav/h(i,jmm1))*(el(i,jmm1)-eln(i))
            vaf(i,jm)=ramp*vaf(i,jm)
            uaf(i,jm)=0.e0
          end if
        end do

        do j=1,jm
          do i=1,im
            uaf(i,j)=uaf(i,j)*dum(i,j)
            vaf(i,j)=vaf(i,j)*dvm(i,j)
          end do
        end do

        return
*/
	else if (idx == 2){

#ifndef TIME_DISABLE
	struct timeval start_bcond_2,
				   end_bcond_2;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_2);
#endif

		/*
		if (n_east == -1){
			for (j = 1; j < jmm1; j++){
				uaf[j][im-1] = uabe[j]+rfe
									  *sqrtf(grav/h[j][imm1-1])
									  *(el[j][imm1-1]-ele[j]);
				uaf[j][im-1] = ramp*uaf[j][im-1];
				vaf[j][im-1] = 0;

			}
		}
		if (n_west == -1){
			for (j = 1; j < jmm1; j++){
				uaf[j][1] = uabw[j]-rfw*sqrtf(grav/h[j][1])
									   *(el[j][1]-elw[j]);		
				uaf[j][1] = ramp*uaf[j][1];
				uaf[j][0] = uaf[j][1];
				vaf[j][0] = 0;
			}
		}

		if (n_north == -1){
			for (i = 1; i < imm1; i++){
				vaf[jm-1][i] = vabn[i]+rfn
									  *sqrtf(grav/h[jmm1-1][i])	
									  *(el[jmm1-1][i]-eln[i]);
				vaf[jm-1][i] = ramp*vaf[jm-1][i];
				uaf[jm-1][i] = 0;
			}
		}

		if (n_south == -1){
			for (i = 1; i < imm1; i++){
				vaf[1][i] = vabs[i]-rfs*sqrtf(grav/h[1][i])
									   *(el[1][i]-els[i]);
				vaf[1][i] = ramp*vaf[1][i];
				vaf[0][i] = vaf[1][i];
				uaf[0][i] = 0;
			}
		}

		if (iperx != 0){
			xperi2d_mpi(uaf, im, jm);	
			xperi2d_mpi(vaf, im, jm);	
			if (iperx < 0){
				if (n_north == -1){
					for (i = 0; i < im; i++){
						uaf[jm-1][i] = uaf[jmm1-1][i];
						dum[jm-1][i] = 1.f;
					}
				}
				if (n_south == -1){
					for (i = 0; i < im; i++){
						uaf[0][i] = uaf[1][i];
						dum[0][i] = 1.f;
					}
				}
			}
		}

		if (ipery != 0){
			yperi2d_mpi(uaf, im, jm);	
			yperi2d_mpi(vaf, im, jm);	
			if (ipery < 0){
				if (n_east == -1){
					for (j = 0; j < jm; j++){
						vaf[j][im-1] = vaf[j][imm1-1];	
						dvm[j][im-1] = 1.f;
					}
				}

				if (n_west == -1){
					for (j = 0; j < jm; j++){
						vaf[j][0] = vaf[j][1];
						dvm[j][0] = 1.f;
					}
				}
			}
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uaf[j][i] = uaf[j][i]*dum[j][i];
				vaf[j][i] = vaf[j][i]*dvm[j][i];
			}
		}
		*/

		/*
	
		checkCudaErrors(cudaMemcpy(d_uaf, uaf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vaf, vaf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_el, el, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

		bcond_gpu_kernel_2_0<<<blockPerGrid, threadPerBlock>>>(
				d_uaf, d_vaf, d_el, d_uabe, d_uabw, d_vabs, d_vabn, 
				d_ele, d_elw, d_els, d_eln, d_h, 
				rfe, rfw, rfs, rfn, ramp, grav, 
				n_east, n_west, n_south, n_north, jm, im);

		if (iperx != 0){
			//xperi2d_mpi_gpu(d_uaf, im, jm);
			//xperi2d_mpi_gpu(d_vaf, im, jm);
			xperi2d_cuda_aware_mpi(d_uaf, im, jm);
			xperi2d_cuda_aware_mpi(d_vaf, im, jm);

			if (iperx < 0){
				bcond_gpu_kernel_2_1<<<blockPerGrid, threadPerBlock>>>(
						d_uaf, d_dum, n_north, n_south, jm, im);
			}
		}

		if (ipery != 0){
			yperi2d_mpi_gpu(d_uaf, im, jm);
			yperi2d_mpi_gpu(d_vaf, im, jm);

			if (ipery < 0){
				bcond_gpu_kernel_2_2<<<blockPerGrid, threadPerBlock>>>(
						d_vaf, d_dvm, n_east, n_west, jm, im);
			}
		}


		bcond_gpu_kernel_2_3<<<blockPerGrid, threadPerBlock>>>(
				d_uaf, d_vaf, d_dum, d_dvm, jm, im);


		/*
		checkCudaErrors(cudaMemcpy(uaf, d_uaf, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vaf, d_vaf, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/
		//modify +uaf, +vaf (because we do not modify each element, 
		//but now we have to copy-back every element)
		/*
		bcond_gpu_kernel_2(float *uaf, float *vaf, float *el,
						   float *uabe, float *uabw, 
						   float *vabs, float *vabn, 
						   float *ele, float *elw,
						   float *els, float *eln,
						   float *h, 
						   float *dum, float *dvm,
						   float rfe, float rfw, 
						   float rfs, float rfn,
						   float ramp, float grav,  
						   int n_east, int n_west,
						   int n_south, int n_north,
						   int jm, int im){

		*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_2);
		bcond_time_2 += time_consumed(&start_bcond_2, 
								      &end_bcond_2);
#endif
		//return;
	}
	/*
      else if(idx.eq.3) then

! internal (3-D) velocity boundary conditions

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              uf(i,j,k)=uf(i,j,k)*dum(i,j)
              vf(i,j,k)=vf(i,j,k)*dvm(i,j)
            end do
          end do
        end do

        return

	*/
	
	else if(idx == 3){

#ifndef TIME_DISABLE
	struct timeval start_bcond_3,
				   end_bcond_3;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_3);
#endif
		/*
//!     EAST
//!     radiation boundary conditions.
		if (n_east == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					ga = sqrtf(h[j][im-1]/hmax);
					uf[k][j][im-1] = ga*(0.25f*u[k][j-1][imm1-1]
										+0.5f*u[k][j][imm1-1]
										+0.25f*u[k][j+1][imm1-1])
									+(1.f-ga)*(0.25f*u[k][j-1][im-1]
											  +0.5f*u[k][j][im-1]
											  +0.25f*u[k][j+1][im-1]);
					vf[k][j][im-1] = 0;
				}
			}
		}

//!     WEST
//!     radiation boundary conditions.
		if (n_west == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					ga = sqrtf(h[j][0]/hmax);
					uf[k][j][1] = ga*(0.25f*u[k][j-1][2]
									 +0.5f*u[k][j][2]
									 +0.25f*u[k][j+1][2])
								 +(1.f-ga)*(0.25f*u[k][j-1][1]
										   +0.5f*u[k][j][1]
										   +0.25f*u[k][j+1][1]);
					uf[k][j][0] = uf[k][j][1];
					vf[k][j][0] = 0;
				}
			}
		}

//!     NORTH
//!     radiation boundary conditions.
		
		if (n_north == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 1; i < imm1; i++){
					ga = sqrtf(h[jm-1][i]/hmax);
					vf[k][jm-1][i] = ga*(0.25f*v[k][jmm1-1][i-1]
										+0.5f*v[k][jmm1-1][i]
										+0.25f*v[k][jmm1-1][i+1])
									+(1.f-ga)*(0.25f*v[k][jm-1][i-1]
											  +0.5f*v[k][jm-1][i]
											  +0.25f*v[k][jm-1][i+1]);
					uf[k][jm-1][i] = 0;
				}
			}
		}

//!     SOUTH
//!     radiation boundary conditions.
		
		if (n_south == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 1; i < imm1; i++){
					ga = sqrtf(h[0][i]/hmax);	
					vf[k][1][i] = ga*(0.25f*v[k][2][i-1]
									 +0.5f*v[k][2][i]
									 +0.25*v[k][2][i+1])
								 +(1.f-ga)*(0.25f*v[k][1][i-1]
										   +0.5f*v[k][1][i]
										   +0.25f*v[k][1][i+1]);
					vf[k][0][i] = vf[k][1][i];
					uf[k][0][i] = 0;
				}
			}
		}

		if (iperx != 0){
			xperi2d_mpi(wubot, im, jm);
			xperi2d_mpi(wvbot, im, jm);
			xperi3d_mpi(uf, im, jm, kbm1);
			xperi3d_mpi(vf, im, jm, kbm1);

			if (iperx < 0){
				if (n_north == -1){
					for (i = 0; i < im; i++){
						wubot[jm-1][i] = wubot[jmm1-1][i];	
					}
					for (k = 0; k < kbm1; k++){
						for (i = 0; i < im; i++){
							uf[k][jm-1][i] = uf[k][jmm1-1][i];	
						}
					}
				}

				if (n_south == -1){
					for (i = 0; i < im; i++){
						wubot[0][i] = wubot[1][i];	
					}
					for (k = 0; k < kbm1; k++){
						for (i = 0; i < im; i++){
							uf[k][0][i] = uf[k][1][i];	
						}
					}
				}
				
			}

		}

		if (ipery != 0){
			yperi2d_mpi(wubot, im, jm);
			yperi2d_mpi(wvbot, im, jm);
			yperi3d_mpi(uf, im, jm, kbm1);
			yperi3d_mpi(vf, im, jm, kbm1);

			if (ipery < 0){
				if (n_east == -1){
					for (j = 0; j < jm; j++){
						wvbot[j][im-1] = wvbot[j][imm1-1];	
					}
					for (k = 0; k < kbm1; k++){
						for (j = 0; j < jm; j++){
							vf[k][j][im-1] = vf[k][j][imm1-1];	
						}
					}
				}

				if (n_west == -1){
					for (j = 0; j < jm; j++){
						wvbot[j][0] = wvbot[j][1];	
					}
					for (k = 0; k < kbm1; k++){
						for (j = 0; j < jm; j++){
							vf[k][j][0] = vf[k][j][1];	
						}
					}
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*dum[j][i];	
					vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
				}
			}
		}
		*/

		bcond_gpu_kernel_3_0<<<blockPerGrid, threadPerBlock>>>(
				d_uf, d_vf, d_u, d_v, 
				d_h, n_east, n_west, n_south, n_north,
				kb, jm, im);

		if (iperx != 0){
			//xperi2d_mpi_gpu(d_wubot, im, jm);
			//xperi2d_mpi_gpu(d_wvbot, im, jm);
			xperi2d_cuda_aware_mpi(d_wubot, im, jm);
			xperi2d_cuda_aware_mpi(d_wvbot, im, jm);
			//xperi3d_mpi_gpu(d_uf, im, jm, kbm1);
			//xperi3d_mpi_gpu(d_vf, im, jm, kbm1);
			xperi3d_cuda_aware_mpi(d_uf, im, jm, kbm1);
			xperi3d_cuda_aware_mpi(d_vf, im, jm, kbm1);

			if (iperx < 0){
				if (n_north == -1){
					checkCudaErrors(cudaMemcpy(d_wubot+(jm-1)*im,
											   d_wubot+(jmm1-1)*im,
											   im*sizeof(float),
											   cudaMemcpyDeviceToDevice));
					checkCudaErrors(cudaMemcpy2D(d_uf+(jm-1)*im,
												jm*im*sizeof(float),
												d_uf+(jmm1-1)*im,
												jm*im*sizeof(float),
												im*sizeof(float),
												kbm1,
												cudaMemcpyDeviceToDevice));
					//for (i = 0; i < im; i++){
					//	wubot[jm-1][i] = wubot[jmm1-1][i];	
					//}
					//for (k = 0; k < kbm1; k++){
					//	for (i = 0; i < im; i++){
					//		uf[k][jm-1][i] = uf[k][jmm1-1][i];	
					//	}
					//}
				}

				checkCudaErrors(cudaDeviceSynchronize());

				if (n_south == -1){
					checkCudaErrors(cudaMemcpy(d_wubot,
											   d_wubot+im,
											   im*sizeof(float),
											   cudaMemcpyDeviceToDevice));
					
					checkCudaErrors(cudaMemcpy2D(d_uf,
												jm*im*sizeof(float),
												d_uf+im,
												jm*im*sizeof(float),
												im*sizeof(float),
												kbm1,
												cudaMemcpyDeviceToDevice));
					//for (i = 0; i < im; i++){
					//	wubot[0][i] = wubot[1][i];	
					//}
					//for (k = 0; k < kbm1; k++){
					//	for (i = 0; i < im; i++){
					//		uf[k][0][i] = uf[k][1][i];	
					//	}
					//}
				}
			}
		}

		if (ipery != 0){
			yperi2d_mpi_gpu(d_wubot, im, jm);
			yperi2d_mpi_gpu(d_wvbot, im, jm);
			yperi3d_mpi_gpu(d_uf, im, jm, kbm1);
			yperi3d_mpi_gpu(d_vf, im, jm, kbm1);

			if (ipery < 0){
				if (n_east == -1){
					//for (j = 0; j < jm; j++){
					//	wvbot[j][im-1] = wvbot[j][imm1-1];	
					//}
					//for (k = 0; k < kbm1; k++){
					//	for (j = 0; j < jm; j++){
					//		vf[k][j][im-1] = vf[k][j][imm1-1];	
					//	}
					//}
					checkCudaErrors(cudaMemcpy2D(d_wvbot+im-1,
												 im*sizeof(float),
												 d_wvbot+imm1-1,
												 im*sizeof(float),
												 sizeof(float),
												 jm,
												 cudaMemcpyDeviceToDevice));

					checkCudaErrors(cudaMemcpy2D(d_vf+im-1,
												 im*sizeof(float),
												 d_vf+imm1-1,
												 im*sizeof(float),
												 sizeof(float),
												 kbm1*jm,
												 cudaMemcpyDeviceToDevice));

				}

				if (n_west == -1){
					//for (j = 0; j < jm; j++){
					//	wvbot[j][0] = wvbot[j][1];	
					//}
					//for (k = 0; k < kbm1; k++){
					//	for (j = 0; j < jm; j++){
					//		vf[k][j][0] = vf[k][j][1];	
					//	}
					//}
					checkCudaErrors(cudaMemcpy2D(d_wvbot,
												 im*sizeof(float),
												 d_wvbot+1,
												 im*sizeof(float),
												 sizeof(float),
												 jm,
												 cudaMemcpyDeviceToDevice));

					checkCudaErrors(cudaMemcpy2D(d_vf,
												 im*sizeof(float),
												 d_vf+1,
												 im*sizeof(float),
												 sizeof(float),
												 kbm1*jm,
												 cudaMemcpyDeviceToDevice));
				}
			}
		}


		bcond_gpu_kernel_3_1<<<blockPerGrid, threadPerBlock>>>(
				d_uf, d_vf, d_dum, d_dvm,
				kb, jm, im);

		/*
		checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

//		bcond_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
//				d_uf, d_vf, d_dum, d_dvm, kb, jm, im);

		/*
		bcond_gpu_kernel_3(float *uf, float *vf,
						   float *dum, float *dvm,
						   int kb, int jm, int im);
		*/
		/*
		checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_3);
		bcond_time_3 += time_consumed(&start_bcond_3, 
								      &end_bcond_3);
#endif
		//return;
	}
	
	/*
      else if(idx.eq.4) then

! temperature and salinity boundary conditions (using uf and vf,
! respectively)
        do k=1,kbm1
          do j=1,jm
            ! east
            if(n_east.eq.-1) then
              u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
              if(u1.le.0.e0) then
                uf(im,j,k)=t(im,j,k)-u1*(tbe(j,k)-t(im,j,k))
                vf(im,j,k)=s(im,j,k)-u1*(sbe(j,k)-s(im,j,k))
              else
                uf(im,j,k)=t(im,j,k)-u1*(t(im,j,k)-t(imm1,j,k))
                vf(im,j,k)=s(im,j,k)-u1*(s(im,j,k)-s(imm1,j,k))
                if(k.ne.1.and.k.ne.kbm1) then
                  wm=.5e0*(w(imm1,j,k)+w(imm1,j,k+1))*dti
     $                /((zz(k-1)-zz(k+1))*dt(imm1,j))
                  uf(im,j,k)=uf(im,j,k)-wm*(t(imm1,j,k-1)-t(imm1,j,k+1))
                  vf(im,j,k)=vf(im,j,k)-wm*(s(imm1,j,k-1)-s(imm1,j,k+1))
                endif
              end if
            end if

            ! west
            if(n_west.eq.-1) then
              u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
              if(u1.ge.0.e0) then
                uf(1,j,k)=t(1,j,k)-u1*(t(1,j,k)-tbw(j,k))
                vf(1,j,k)=s(1,j,k)-u1*(s(1,j,k)-sbw(j,k))
              else
                uf(1,j,k)=t(1,j,k)-u1*(t(2,j,k)-t(1,j,k))
                vf(1,j,k)=s(1,j,k)-u1*(s(2,j,k)-s(1,j,k))
                if(k.ne.1.and.k.ne.kbm1) then
                  wm=.5e0*(w(2,j,k)+w(2,j,k+1))*dti
     $                /((zz(k-1)-zz(k+1))*dt(2,j))
                  uf(1,j,k)=uf(1,j,k)-wm*(t(2,j,k-1)-t(2,j,k+1))
                  vf(1,j,k)=vf(1,j,k)-wm*(s(2,j,k-1)-s(2,j,k+1))
                end if
              end if
            end if
          end do

          do i=1,im
            ! south
            if(n_south.eq.-1) then
              u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
              if(u1.ge.0.e0) then
                uf(i,1,k)=t(i,1,k)-u1*(t(i,1,k)-tbs(i,k))
                vf(i,1,k)=s(i,1,k)-u1*(s(i,1,k)-sbs(i,k))
              else
                uf(i,1,k)=t(i,1,k)-u1*(t(i,2,k)-t(i,1,k))
                vf(i,1,k)=s(i,1,k)-u1*(s(i,2,k)-s(i,1,k))
                if(k.ne.1.and.k.ne.kbm1) then
                  wm=.5e0*(w(i,2,k)+w(i,2,k+1))*dti
     $                /((zz(k-1)-zz(k+1))*dt(i,2))
                  uf(i,1,k)=uf(i,1,k)-wm*(t(i,2,k-1)-t(i,2,k+1))
                  vf(i,1,k)=vf(i,1,k)-wm*(s(i,2,k-1)-s(i,2,k+1))
                end if
              end if
            end if

            ! north
            if(n_north.eq.-1) then
              u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
              if(u1.le.0.e0) then
                uf(i,jm,k)=t(i,jm,k)-u1*(tbn(i,k)-t(i,jm,k))
                vf(i,jm,k)=s(i,jm,k)-u1*(sbn(i,k)-s(i,jm,k))
              else
                uf(i,jm,k)=t(i,jm,k)-u1*(t(i,jm,k)-t(i,jmm1,k))
                vf(i,jm,k)=s(i,jm,k)-u1*(s(i,jm,k)-s(i,jmm1,k))
                if(k.ne.1.and.k.ne.kbm1) then
                  wm=.5e0*(w(i,jmm1,k)+w(i,jmm1,k+1))*dti
     $                /((zz(k-1)-zz(k+1))*dt(i,jmm1))
                  uf(i,jm,k)=uf(i,jm,k)-wm*(t(i,jmm1,k-1)-t(i,jmm1,k+1))
                  vf(i,jm,k)=vf(i,jm,k)-wm*(s(i,jmm1,k-1)-s(i,jmm1,k+1))
                end if
              end if
            end if
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              uf(i,j,k)=uf(i,j,k)*fsm(i,j)
              vf(i,j,k)=vf(i,j,k)*fsm(i,j)
            end do
          end do
        end do

        return

	*/
	
	else if (idx == 4){

#ifndef TIME_DISABLE
	struct timeval start_bcond_4,
				   end_bcond_4;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_4);
#endif

		//only modify uf & vf

		/*
		checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_t, t, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_s, s, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

		bcond_gpu_kernel_4_0<<<blockPerGrid, threadPerBlock>>>(
				d_u, d_uf, d_v, d_vf, d_t, d_s, d_w, d_dt, 
				d_tobe, d_sobe, d_tobw, d_sobw,
				d_tobs, d_sobs, d_tobn, d_sobn,
				d_tbe, d_sbe, d_tbw, d_sbw,
				d_tbs, d_sbs, d_tbn, d_sbn,
				d_dx, d_dy, d_zz, d_frz,
				dti, n_east, n_west, n_south, n_north, kb, jm, im);

		if (iperx != 0){
			//xperi3d_mpi_gpu(d_uf, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_vf, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_uf, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_vf, im, jm, kbm1);	
		}

		if (ipery != 0){
			yperi3d_mpi_gpu(d_uf, im, jm, kbm1);	
			yperi3d_mpi_gpu(d_vf, im, jm, kbm1);	
		}

		bcond_gpu_kernel_4_1<<<blockPerGrid, threadPerBlock>>>(
				d_uf, d_vf, d_fsm,
				kb, jm, im);

		/*
		bcond_gpu_kernel_4(float *u, float *uf,
				   float *v, float *vf, 
				   float *t, float *s, 
				   float *w, float *dt, 
				   float *dx, float *dy,
				   float *tbe, float *sbe, 
				   float *tbw, float *sbw,
				   float *tbs, float *sbs,
				   float *tbn, float *sbn,
				   float *zz, float *fsm,
				   float dti, 
				   int n_east, int n_west, 
				   int n_south, int n_north,
				   int kb, int jm, int im){
		*/

		/*
		checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_4);
		bcond_time_4 += time_consumed(&start_bcond_4, 
								      &end_bcond_4);
#endif

		//return;
	}

	/*
      else if(idx.eq.5) then

! vertical velocity boundary conditions
        do k=1,kbm1
          do j=1,jm
            do i=1,im
              w(i,j,k)=w(i,j,k)*fsm(i,j)
            end do
          end do
        end do

        return
	*/
	
	else if (idx == 5){

#ifndef TIME_DISABLE
	struct timeval start_bcond_5,
				   end_bcond_5;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_5);
#endif
		/*
		checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/
		if (iperx != 0){
			//xperi3d_mpi_gpu(d_w, im, jm, kbm1);
			xperi3d_cuda_aware_mpi(d_w, im, jm, kbm1);
		}

		if (ipery != 0){
			yperi3d_mpi_gpu(d_w, im, jm, kbm1);	
		}

		bcond_gpu_kernel_5<<<blockPerGrid, threadPerBlock>>>(
				d_w, d_fsm, kb, jm, im);

//		bcond_gpu_kernel_5<<<blockPerGrid, threadPerBlock>>>(
//				d_w, d_fsm, kb, jm, im);

		/*
		checkCudaErrors(cudaMemcpy(w, d_w, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/
		/*
		bcond_gpu_kernel_5(float *w, float *fsm,
				   int kb, int jm, int im){
		*/

		/*
		if (iperx != 0){
			xperi3d_mpi(w, im, jm, kbm1);
		}

		if (ipery != 0){
			yperi3d_mpi(w, im, jm, kbm1);
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					w[k][j][i] *= fsm[j][i];	
				}
			}
		}
		*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_5);
		bcond_time_5 += time_consumed(&start_bcond_5, 
								      &end_bcond_5);
#endif
		//return;
	}


	/*
      else if(idx.eq.6) then

! q2 and q2l boundary conditions

        do k=1,kb
          do j=1,jm
            ! west
            if(n_west.eq.-1) then
              u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
              if(u1.ge.0.e0) then
                uf(1,j,k)=q2(1,j,k)-u1*(q2(1,j,k)-small)
                vf(1,j,k)=q2l(1,j,k)-u1*(q2l(1,j,k)-small)
              else
                uf(1,j,k)=q2(1,j,k)-u1*(q2(2,j,k)-q2(1,j,k))
                vf(1,j,k)=q2l(1,j,k)-u1*(q2l(2,j,k)-q2l(1,j,k))
              end if
            end if

            ! east
            if(n_east.eq.-1) then
              u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
              if(u1.le.0.e0) then
                uf(im,j,k)=q2(im,j,k)-u1*(small-q2(im,j,k))
                vf(im,j,k)=q2l(im,j,k)-u1*(small-q2l(im,j,k))
              else
                uf(im,j,k)=q2(im,j,k)-u1*(q2(im,j,k)-q2(imm1,j,k))
                vf(im,j,k)=q2l(im,j,k)-u1*(q2l(im,j,k)-q2l(imm1,j,k))
              end if
            end if
          end do

          do i=1,im
            ! south
            if(n_south.eq.-1) then
              u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
              if(u1.ge.0.e0) then
                uf(i,1,k)=q2(i,1,k)-u1*(q2(i,1,k)-small)
                vf(i,1,k)=q2l(i,1,k)-u1*(q2l(i,1,k)-small)
              else
                uf(i,1,k)=q2(i,1,k)-u1*(q2(i,2,k)-q2(i,1,k))
                vf(i,1,k)=q2l(i,1,k)-u1*(q2l(i,2,k)-q2l(i,1,k))
              end if
            end if

            ! north
            if(n_north.eq.-1) then
              u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
              if(u1.le.0.e0) then
                uf(i,jm,k)=q2(i,jm,k)-u1*(small-q2(i,jm,k))
                vf(i,jm,k)=q2l(i,jm,k)-u1*(small-q2l(i,jm,k))
              else
                uf(i,jm,k)=q2(i,jm,k)-u1*(q2(i,jm,k)-q2(i,jmm1,k))
                vf(i,jm,k)=q2l(i,jm,k)-u1*(q2l(i,jm,k)-q2l(i,jmm1,k))
              end if
            end if
          end do
        end do

        do k=1,kb
          do j=1,jm
            do i=1,im
              uf(i,j,k)=uf(i,j,k)*fsm(i,j)+1.e-10
              vf(i,j,k)=vf(i,j,k)*fsm(i,j)+1.e-10
            end do
          end do
        end do

        return

      end if
	  
	*/
	
	else if (idx == 6){

#ifndef TIME_DISABLE
	struct timeval start_bcond_6,
				   end_bcond_6;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_6);
#endif
		//modify uf vf
		/*
		if (n_east == -1){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][im-1]*dti
						/(dx[j][im-1]+dx[j][imm1-1]);	
					if (u1 <= 0.0f){
						uf[k][j][im-1] = q2[k][j][im-1]-u1*(small-q2[k][j][im-1]);
						vf[k][j][im-1] = q2l[k][j][im-1]-u1*(small-q2l[k][j][im-1]);
					}else{
						uf[k][j][im-1] = q2[k][j][im-1]
							         -u1*(q2[k][j][im-1]-q2[k][j][imm1-1]);	

						vf[k][j][im-1] = q2l[k][j][im-1]
							         -u1*(q2l[k][j][im-1]-q2l[k][j][imm1-1]);
					}
				}
			}
		}

		if (n_west == -1){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][1]*dti/(dx[j][0]+dx[j][1]);	
					if (u1 >= 0.0f){
						uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][0]-small);
						vf[k][j][0] = q2l[k][j][0]-u1*(q2l[k][j][0]-small);
					}else{
						uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][1]-q2[k][j][0]);	
						vf[k][j][0] = q2l[k][j][0] - u1*(q2l[k][j][1]-q2l[k][j][0]);
					}
				}
			}
		}

		if (n_north == -1){
			for (k = 0; k < kb; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][jm-1][i]*dti/(dy[jm-1][i]+dy[jmm1-1][i]);		
					if (u1 <= 0.0f){
						uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(small-q2[k][jm-1][i]);	
						vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(small-q2l[k][jm-1][i]);
					}else{
						uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(q2[k][jm-1][i]-q2[k][jmm1-1][i]);	
						vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(q2l[k][jm-1][i]-q2l[k][jmm1-1][i]);
					}

				}
			}
		}

		if (n_south == -1){
			for (k = 0; k < kb; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][1][i]*dti/(dy[0][i]+dy[1][i]);	
					if (u1 >= 0.0f){
						uf[k][0][i] = q2[k][0][i]-u1*(q2[k][0][i]-small);	
						vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][0][i]-small);
					}else{
						uf[k][0][i] = q2[k][0][i]-u1*(q2[k][1][i]-q2[k][0][i]);	
						vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][1][i]-q2l[k][0][i]);
					}
				}
			}
		}
		*/
		bcond_gpu_kernel_6_0<<<blockPerGrid, threadPerBlock>>>(
				d_u, d_v, d_uf, d_vf, d_q2, d_q2l, 
				d_dx, d_dy, 
				dti, small, n_east, n_west, n_south, n_north, 
				kb, jm, im);
		
		if (iperx != 0){
			//xperi3d_mpi_gpu(d_uf, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_vf, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_kh, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_km, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_kq, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_l, im, jm, kbm1);	

			xperi3d_cuda_aware_mpi(d_uf, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_vf, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_kh, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_km, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_kq, im, jm, kbm1);	
			xperi3d_cuda_aware_mpi(d_l, im, jm, kbm1);	
		}

		if (ipery != 0){
			yperi3d_mpi_gpu(d_uf, im, jm, kbm1);
			yperi3d_mpi_gpu(d_vf, im, jm, kbm1);
			yperi3d_mpi_gpu(d_kh, im, jm, kbm1);
			yperi3d_mpi_gpu(d_km, im, jm, kbm1);
			yperi3d_mpi_gpu(d_kq, im, jm, kbm1);
			yperi3d_mpi_gpu(d_l, im, jm, kbm1);
		}

		/*
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*fsm[j][i];
					vf[k][j][i] = vf[k][j][i]*fsm[j][i];
				}
			}
		}
		*/

		bcond_gpu_kernel_6_1<<<blockPerGrid, threadPerBlock>>>(
				d_uf, d_vf, d_fsm,
				kb, jm, im);

		/*
		checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_q2, q2, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_q2l, q2l, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

//		bcond_gpu_kernel_6<<<blockPerGrid, threadPerBlock>>>(
//				d_u, d_v, d_uf, d_vf, d_q2, d_q2l, d_dx, d_dy, d_fsm,
//				dti, small, n_east, n_west, n_south, n_north, kb, jm, im);
		/*
		bcond_gpu_kernel_6(float *u, float *v, 
				   float *uf, float *vf,
				   float *q2, float *q2l,
				   float *dx, float *dy,
				   float dti, float small,
				   int n_east, int n_west, 
				   int n_south, int n_north,
				   int kb, int jm, int im){
		*/

		/*
		checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_6);
		bcond_time_6 += time_consumed(&start_bcond_6, 
								      &end_bcond_6);
#endif
		//return;
	}

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&end_bcond);
	bcond_time += time_consumed(&start_bcond, 
							    &end_bcond);
#endif
	
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_1_0(float *elf,
						      int n_west, int n_east,
							  int jm, int im){

	const int j = blockIdx.y*blockDim.y+threadIdx.y;

	if (n_west == -1){
		if (j < jm){
			elf[j_off] = elf[j_off+1];	
		}
	}

	if (n_east == -1){
		if (j < jm){
			elf[j_off+im-1] = elf[j_off+im-2];
		}
	}
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_1_0(float *elf,
							  int n_north, int n_south,
							  int jm, int im){

	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int jmm1 = jm-1;

	if (n_north == -1){
		if (i < im){
			elf[jm_1_off+i] = elf[jmm1_1_off+i];	
		}
	}

	if (n_south == -1){
		if (i < im){
			elf[i] = elf[im+i];
		}
	}
}

__global__ void
bcond_overlap_ew_gpu_kernel_1_1(
				   float * __restrict__ elf, 
				   const float * __restrict__ fsm,
				   int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			elf[j][i] *= fsm[j][i];	
		}
	}
	*/

	if (j < jm-1){
		elf[j_off+i] *= fsm[j_off+i];	
	}
	return;
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_1_1(
				   float * __restrict__ elf, 
				   const float * __restrict__ fsm,
				   int n_east, int n_west,
				   int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y; 

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			elf[j][i] *= fsm[j][i];	
		}
	}
	*/

	if (n_east == -1){
		if (j < jm){
			elf[j_off+im-1] *= fsm[j_off+im-1];	
		}
	
	}

	if (n_west == -1){
		if (j < jm){
			elf[j_off] *= fsm[j_off];	
		}
	}

	return;
}

__global__ void
bcond_overlap_sn_gpu_kernel_1_1(
				   float * __restrict__ elf, 
				   const float * __restrict__ fsm,
				   int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			elf[j][i] *= fsm[j][i];	
		}
	}
	*/

	if (i > 32 && i < im-33){
		elf[j_off+i] *= fsm[j_off+i];	
	}
	return;
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_1_1(
				   float * __restrict__ elf, 
				   const float * __restrict__ fsm,
				   int n_south, int n_north,
				   int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x; 

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			elf[j][i] *= fsm[j][i];	
		}
	}
	*/

	if (n_south == -1){
		if (i > 0 && i < im-1){
			elf[i] *= fsm[i];	
		}
	}
	if (n_north == -1){
		if (i > 0 && i < im-1){
			elf[jm_1_off+i] *= fsm[jm_1_off+i];	
		}
	}
	return;
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_2_0(
				   float *uaf, float *vaf, float *el,
				   float *uabe, float *uabw, 
				   float *ele, float *elw,
				   float *h, 
				   float rfe, float rfw, 
				   float ramp, float grav,  
				   int n_east, int n_west,
				   int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 
	const int jmm1 = jm-1; 
	const int imm1 = im-1; 

	if (n_east == -1){
		if (j > 0 && j < jmm1){
			uaf[j_off+(im-1)] = uabe[j]
							   +rfe*sqrtf(grav/h[j_off+(imm1-1)])
								   *(el[j_off+(imm1-1)]-ele[j]);
			uaf[j_off+(im-1)] *= ramp;
			vaf[j_off+(im-1)] = 0;
		}
	}

	if (n_west == -1){
		if (j < jmm1){
			uaf[j_off+1] = uabw[j]
						  -rfw*sqrtf(grav/h[j_off+1])
							  *(el[j_off+1]-elw[j]);
			uaf[j_off+1] *= ramp;
			uaf[j_off] = uaf[j_off+1];
			vaf[j_off] = 0;
		}
	}
	return;
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_2_0(
				   float *uaf, float *vaf, float *el,
				   float *vabs, float *vabn, 
				   float *els, float *eln,
				   float *h, 
				   float rfs, float rfn,
				   float ramp, float grav,  
				   int n_south, int n_north,
				   int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	const int jmm1 = jm-1; 
	const int imm1 = im-1; 

	if (n_north == -1){
		if (i < imm1){
			vaf[jm_1_off+i] = vabn[i]
							 +rfn*sqrtf(grav/h[jmm1_1_off+i])
								 *(el[jmm1_1_off+i]-eln[i]);
			vaf[jm_1_off+i] *= ramp;
			uaf[jm_1_off+i] = 0;
		}
	}
	
	if (n_south == -1){
		if (i < imm1){
			vaf[im+i] = vabs[i]
					   -rfs*sqrtf(grav/h[im+i])
						   *(el[im+i]-els[i]);
			vaf[im+i] *= ramp;
			vaf[i] = vaf[im+i];
			uaf[i] = 0;
		}
	}

	return;
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_2_1(
					 float *uaf, float *dum,
					 int n_north, int n_south,
				     int jm, int im){
	
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 
	const int jmm1 = jm-1; 

	if (n_north == -1){
		if (i < im){
			uaf[jm_1_off+i] = uaf[jmm1_1_off+i];
			dum[jm_1_off+i] = 1.f;
		}
	}
	if (n_south == -1){
		if (i < im){
			uaf[i] = uaf[im+i];
			dum[i] = 1.f;
		}
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_2_2(
					 float *vaf, float *dvm,
					 int n_east, int n_west,
				     int jm, int im){
	
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 
	const int imm1 = im-1; 

	if (n_east == -1){
		if (j < jm){
			vaf[j_off+im-1] = vaf[j_off+imm1-1];
			dvm[j_off+im-1] = 1.f;
		}
	}
	if (n_west == -1){
		if (j < jm){
			vaf[j_off] = vaf[j_off+1];
			dvm[j_off] = 1.f;
		}
	}
}

__global__ void
bcond_overlap_ew_gpu_kernel_2_3(
					 float *uaf, float *vaf,
				     float *dum, float *dvm,
				     int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}
	
	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			uaf[j][i] *= dum[j][i];	
			vaf[j][i] *= dvm[j][i];
		}
	}
	*/

	if (j < jm-1){
		uaf[j_off+i] *= dum[j_off+i];
		vaf[j_off+i] *= dvm[j_off+i];
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_2_3(
					 float *uaf, float *vaf,
				     float *dum, float *dvm,
					 int n_east, int n_west,
				     int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y; 

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			uaf[j][i] *= dum[j][i];	
			vaf[j][i] *= dvm[j][i];
		}
	}
	*/

	if (n_east == -1){
		if (j < jm){
			uaf[j_off+im-1] *= dum[j_off+im-1];
			vaf[j_off+im-1] *= dvm[j_off+im-1];
		}
	}

	if (n_west == -1){
		if (j < jm){
			uaf[j_off] *= dum[j_off];
			vaf[j_off] *= dvm[j_off];
		}
	}
}

__global__ void
bcond_overlap_sn_gpu_kernel_2_3(
					 float *uaf, float *vaf,
				     float *dum, float *dvm,
					 int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}
	
	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			uaf[j][i] *= dum[j][i];	
			vaf[j][i] *= dvm[j][i];
		}
	}
	*/

	if (i > 32 && i < im-33){
		uaf[j_off+i] *= dum[j_off+i];
		vaf[j_off+i] *= dvm[j_off+i];
	}
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_2_3(
					 float *uaf, float *vaf,
				     float *dum, float *dvm,
				     int n_south, int n_north, 
					 int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	
	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			uaf[j][i] *= dum[j][i];	
			vaf[j][i] *= dvm[j][i];
		}
	}
	*/

	if (n_south == -1){
		if (i < im-1){
			uaf[i] *= dum[i];
			vaf[i] *= dvm[i];
		}
	}

	if (n_north == -1){
		if (i < im-1){
			uaf[jm_1_off+i] *= dum[jm_1_off+i];
			vaf[jm_1_off+i] *= dvm[jm_1_off+i];
		}
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_3_0(
					 float *uf, float *vf,
					 float *u, 
					 float *h,
					 int n_east, int n_west,
					 int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y;

	int imm1 = im-1;
	int jmm1 = jm-1;
	int kbm1 = kb-1;

	float ga;
	const float hmax = 8000.f;

	if (n_east == -1){
		if (j > 0 && j < jmm1){
			for (k = 0; k < kbm1; k++){
				ga = sqrtf(h[j_off+im-1]/hmax);
				uf[k_off+j_off+im-1] = ga*(0.25f*u[k_off+j_1_off+imm1-1]
										  +0.5f*u[k_off+j_off+imm1-1]
										  +0.25f*u[k_off+j_A1_off+imm1-1])
									  +(1.f-ga)
										*(0.25f*u[k_off+j_1_off+im-1]
										 +0.5f*u[k_off+j_off+im-1]
										 +0.25f*u[k_off+j_A1_off+im-1]);

				vf[k_off+j_off+im-1] = 0;
			}
		}
	}

	if (n_west == -1){
		if (j > 0 && j < jmm1){
			for (k = 0; k < kbm1; k++){
				ga = sqrtf(h[j_off]/hmax);
				uf[k_off+j_off+1] = ga*(0.25f*u[k_off+j_1_off+2]
									   +0.5f*u[k_off+j_off+2]
									   +0.25f*u[k_off+j_A1_off+2])
								   +(1.f-ga)
									  *(0.25f*u[k_off+j_1_off+1]
									   +0.5f*u[k_off+j_off+1]
									   +0.25f*u[k_off+j_A1_off+1]);

				uf[k_off+j_off] = uf[k_off+j_off+1];
				vf[k_off+j_off] = 0;
			}
		}
	}
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_3_0(
					 float *uf, float *vf,
					 float *v,
					 float *h,
					 int n_south, int n_north,
					 int kb, int jm, int im){

	int k;
	const int i = blockIdx.x*blockDim.x+threadIdx.x;

	int imm1 = im-1;
	int jmm1 = jm-1;
	int kbm1 = kb-1;

	float ga;
	const float hmax = 8000.f;

	if (n_south == -1){
		if (i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				ga = sqrtf(h[i]/hmax);
				vf[k_off+im+i] = ga*(0.25f*v[k_off+2*im+i-1]
									+0.5f*v[k_off+2*im+i]
									+0.25f*v[k_off+2*im+i+1])
								+(1.f-ga)
								   *(0.25f*v[k_off+im+i-1]
									+0.5f*v[k_off+im+i]
									+0.25f*v[k_off+im+i+1]);
				vf[k_off+i] = vf[k_off+im+i];
				uf[k_off+i] = 0;
			}
		}
	}

	if (n_north == -1){
		if (i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				ga = sqrtf(h[jm_1_off+i]/hmax);	
				vf[k_off+jm_1_off+i] = ga*(0.25f*v[k_off+jmm1_1_off+i-1]
										  +0.5f*v[k_off+jmm1_1_off+i]
										  +0.25f*v[k_off+jmm1_1_off+i+1])
									  +(1.f-ga)
										 *(0.25f*v[k_off+jm_1_off+i-1]
										  +0.5f*v[k_off+jm_1_off+i]
										  +0.25f*v[k_off+jm_1_off+i+1]);
				uf[k_off+jm_1_off+i] = 0;
			}
		}
	}

}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_3_1(
					 float * __restrict__ wubot, 
					 float * __restrict__ uf,
					 float * __restrict__ vf,
					 int n_south, int n_north,
					 int kb, int jm, int im){

	int k;
	const int i = blockIdx.x*blockDim.x+threadIdx.x;

	int jmm1 = jm-1;

	if (n_north == -1){
		if (i < im){
			//wubot[jm-1][i] = wubot[jmm1-1][i];	
			wubot[jm_1_off+i] = wubot[jmm1_1_off+i];	

			for (k = 0; k < kb-1; k++){
				//uf[k][jm-1][i] = uf[k][jmm1-1][i];	
				uf[k_off+jm_1_off+i] = uf[k_off+jmm1_1_off+i];	
			}
		}
	}

	if (n_south == -1){
		if (i < im){
			//wubot[0][i] = wubot[1][i];	
			wubot[i] = wubot[im+i];	
			for (k = 0; k < kb-1; k++){
				uf[k_off+i] = uf[k_off+im+i];	
			}
		}
	}
}

__global__ void
bcond_overlap_ew_gpu_kernel_3_2(float *uf, float *vf,
				     float *dum, float *dvm,
				     int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = uf[k][j][i]*dum[j][i];	
				vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
			}
		}
	}
	*/

	if (j < jm-1){
		for (k = 0; k < kb-1; k++){
			uf[k_off+j_off+i] *= dum[j_off+i];
			vf[k_off+j_off+i] *= dvm[j_off+i];
		}
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_3_2(
					 float *uf, float *vf,
				     float *dum, float *dvm,
					 int n_east, int n_west,
				     int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = uf[k][j][i]*dum[j][i];	
				vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
			}
		}
	}
	*/

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kb-1; k++){
				uf[k_off+j_off+im-1] *= dum[j_off+im-1];
				vf[k_off+j_off+im-1] *= dvm[j_off+im-1];
			}
		}
	}

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kb-1; k++){
				uf[k_off+j_off] *= dum[j_off];
				vf[k_off+j_off] *= dvm[j_off];
			}
		}
	}
}

__global__ void
bcond_overlap_sn_gpu_kernel_3_2(float *uf, float *vf,
				     float *dum, float *dvm,
				     int kb, int jm, int im){

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = uf[k][j][i]*dum[j][i];	
				vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
			}
		}
	}
	*/

	if (i > 32 && i < im-33){
		for (k = 0; k < kb-1; k++){
			uf[k_off+j_off+i] *= dum[j_off+i];
			vf[k_off+j_off+i] *= dvm[j_off+i];
		}
	}
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_3_2(
					 float *uf, float *vf,
				     float *dum, float *dvm,
					 int n_south, int n_north,
				     int kb, int jm, int im){

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = uf[k][j][i]*dum[j][i];	
				vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
			}
		}
	}
	*/

	if (n_south == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb-1; k++){
				uf[k_off+i] *= dum[i];
				vf[k_off+i] *= dvm[i];
			}
		}
	}

	if (n_north == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb-1; k++){
				uf[k_off+jm_1_off+i] *= dum[jm_1_off+i];
				vf[k_off+jm_1_off+i] *= dvm[jm_1_off+i];
			}
		}
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_4_0(float *u, float *uf,
				     float *vf, 
				     float *t, float *s, 
				     float *w, float *dt, 
					 float *tobe, float *sobe,
					 float *tobw, float *sobw,
				     float *tbe, float *sbe, 
				     float *tbw, float *sbw,
				     float *dx, float *dy,
				     float *zz, float *frz,
				     float dti, 
				     int n_east, int n_west, 
				     int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i;
	
	int kbm1 = kb-1;
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	float u1, wm;

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kbm1; k++){
				u1 = 2.0f*u[k_off+j_off+(im-1)]
						 *dti/(dx[j_off+(im-1)]
							  +dx[j_off+(imm1-1)]);
				if (u1 <= 0.0f){
					uf[k_off+j_off+(im-1)] = t[k_off+j_off+(im-1)]
											-u1*(tbe[k*jm+j]
												-t[k_off+j_off+(im-1)]);	

					vf[k_off+j_off+(im-1)] = s[k_off+j_off+(im-1)]
											-u1*(sbe[k*jm+j]
												-s[k_off+j_off+(im-1)]);
				}else{
					uf[k_off+j_off+(im-1)] = t[k_off+j_off+(im-1)]
											-u1*(t[k_off+j_off+(im-1)]
												-t[k_off+j_off+(imm1-1)]);	

					vf[k_off+j_off+(im-1)] = s[k_off+j_off+(im-1)]
											-u1*(s[k_off+j_off+(im-1)]
											    -s[k_off+j_off+(imm1-1)]);

					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+j_off+(imm1-1)]
								  +w[k_A1_off+j_off+(imm1-1)])
								 *dti
								 /((zz[k-1]-zz[k+1])
								  *dt[j_off+(imm1-1)]);

						uf[k_off+j_off+(im-1)] = 
								uf[k_off+j_off+(im-1)]
							   -wm*(t[k_1_off+j_off+(imm1-1)]
								   -t[k_A1_off+j_off+(imm1-1)]);

						vf[k_off+j_off+(im-1)] = 
								vf[k_off+j_off+(im-1)]
							   -wm*(s[k_1_off+j_off+(imm1-1)]
								   -s[k_A1_off+j_off+(imm1-1)]);
					}
				}
			}

			if (nfe > 3){
				for (k = 0; k < kbm1; k++){
					for (i = 0; i < nfe; i++){
						int ii = im-i-1;	
						uf[k_off+j_off+ii] = uf[k_off+j_off+ii]
												*(1.f-frz[j_off+ii])
										   +(tobe[k_off+j_off+i]
												*frz[j_off+ii]);

						vf[k_off+j_off+ii] = vf[k_off+j_off+ii]
												*(1.f-frz[j_off+ii])
											+(sobe[k_off+j_off+i]
												*frz[j_off+ii]);
					}
					//if (j < jm && i >= im-nfe && i < im){
					//	int ii = im-i-1;	
					//	uf[k_off+j_off+i] = uf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i])
					//					   +(tobe[k_off+j_off+ii]
					//							*frz[j_off+i]);

					//	vf[k_off+j_off+i] = vf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i])
					//						+(sobe[k_off+j_off+ii]
					//							*frz[j_off+i]);
					//}
				}
			}
		}
	}

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kbm1; k++){
				u1 = 2.0f*u[k_off+j_off+1]*dti
						 /(dx[j_off]+dx[j_off+1]);
				if (u1 >= 0.0f){
					uf[k_off+j_off] = t[k_off+j_off]
									 -u1*(t[k_off+j_off]
										 -tbw[k*jm+j]);	

					vf[k_off+j_off] = s[k_off+j_off]
									 -u1*(s[k_off+j_off]
										 -sbw[k*jm+j]);

				}else{
					uf[k_off+j_off] = t[k_off+j_off]
									 -u1*(t[k_off+j_off+1]
										 -t[k_off+j_off]);	

					vf[k_off+j_off] = s[k_off+j_off]
									 -u1*(s[k_off+j_off+1]
										 -s[k_off+j_off]);

					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+j_off+1]
								  +w[k_A1_off+j_off+1])
								 *dti
								 /((zz[k-1]-zz[k+1])
								   *dt[j_off+1]);

						uf[k_off+j_off] = uf[k_off+j_off]
										 -wm*(t[k_1_off+j_off+1]
											 -t[k_A1_off+j_off+1]);

						vf[k_off+j_off] = vf[k_off+j_off]
										  -wm*(s[k_1_off+j_off+1]
											  -s[k_A1_off+j_off+1]);
					}
				}
			}

			if (nfw > 3){
				for (k = 0; k < kbm1; k++){
					for (i = 0; i < nfw; i++){
						uf[k_off+j_off+i] = uf[k_off+j_off+i]
												*(1.f-frz[j_off+i])
										   +(tobw[k_off+j_off+i]*frz[j_off+i]);

						vf[k_off+j_off+i] = vf[k_off+j_off+i]
												*(1.f-frz[j_off+i])
										   +(sobw[k_off+j_off+i]*frz[j_off+i]);

					}
					//if (j < jm && i < nfw){
					//	uf[k_off+j_off+i] = uf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i])
					//					   +(tobw[k_off+j_off+i]*frz[j_off+i]);

					//	vf[k_off+j_off+i] = vf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i])
					//					   +(sobw[k_off+j_off+i]*frz[j_off+i]);
					//}
				}
			}
		}
	}
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_4_0(
					 float *uf,
				     float *v, float *vf, 
				     float *t, float *s, 
				     float *w, float *dt, 
					 float *tobs, float *sobs,
					 float *tobn, float *sobn,
				     float *tbs, float *sbs,
				     float *tbn, float *sbn,
				     float *dx, float *dy,
				     float *zz, float *frz,
				     float dti, 
				     int n_south, int n_north,
				     int kb, int jm, int im){

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int j;
	
	int kbm1 = kb-1;
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	float u1, wm;

	if (n_south == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kbm1; k++){
				u1=2.0f*v[k_off+1*im+i]*dti/(dy[i]+dy[1*im+i]);	
				if (u1 >= 0.0f){
					uf[k_off+i] = t[k_off+i]-u1*(t[k_off+i]-tbs[k*im+i]);	
					vf[k_off+i] = s[k_off+i]-u1*(s[k_off+i]-sbs[k*im+i]);
				}else{
					uf[k_off+i] = t[k_off+i]-u1*(t[k_off+1*im+i]-t[k_off+i]);	
					vf[k_off+i] = s[k_off+i]-u1*(s[k_off+1*im+i]-s[k_off+i]);
					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+1*im+i]+w[k_A1_off+1*im+i])
								 *dti
								 /((zz[k-1]-zz[k+1])*dt[1*im+i]);

						uf[k_off+i] = uf[k_off+i]
									 -wm*(t[k_1_off+1*im+i]
										 -t[k_A1_off+1*im+i]);

						vf[k_off+i] = vf[k_off+i]
									 -wm*(s[k_1_off+1*im+i]
										 -s[k_A1_off+1*im+i]);
					}
				}
			}

			if (nfs > 3){
				for (k = 0; k < kbm1; k++){
					for (j = 0; j < nfs; j++){

						uf[k_off+j_off+i] = (uf[k_off+j_off+i]
												*(1.f-frz[j_off+i]))
											+(tobs[k_off+j_off+i]
												*frz[j_off+i]);

						vf[k_off+j_off+i] = (vf[k_off+j_off+i]
												*(1.f-frz[j_off+i]))
											+(sobs[k_off+j_off+i]
												*frz[j_off+i]);

					}

					//if (i < im && j < nfs){
					//	uf[k_off+j_off+i] = (uf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i]))
					//						+(tobs[k_off+j_off+i]
					//							*frz[j_off+i]);

					//	vf[k_off+j_off+i] = (vf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i]))
					//						+(sobs[k_off+j_off+i]
					//							*frz[j_off+i]);
					//}
				}
			}
		}

	}


	if (n_north == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kbm1; k++){
				u1 = 2.0f*v[k_off+jm_1_off+i]
						 *dti
						 /(dy[jm_1_off+i]
						  +dy[jmm1_1_off+i]);	

				if (u1 <= 0){
					uf[k_off+jm_1_off+i] = t[k_off+jm_1_off+i]
										  -u1*(tbn[k*im+i]
											  -t[k_off+jm_1_off+i]);

					vf[k_off+jm_1_off+i] = s[k_off+jm_1_off+i]
										  -u1*(sbn[k*im+i]
											  -s[k_off+jm_1_off+i]);
				}else{
					uf[k_off+jm_1_off+i] = t[k_off+jm_1_off+i]
										  -u1*(t[k_off+jm_1_off+i]
											  -t[k_off+jmm1_1_off+i]);

					vf[k_off+jm_1_off+i] = s[k_off+jm_1_off+i]
										  -u1*(s[k_off+jm_1_off+i]
											  -s[k_off+jmm1_1_off+i]);

					if (k != 0 && k != kbm1-1){
						wm = 0.5f*(w[k_off+jmm1_1_off+i]
								  +w[k_A1_off+jmm1_1_off+i])
								 *dti
								 /((zz[k-1]-zz[k+1])
									*dt[jmm1_1_off+i]);	

						uf[k_off+jm_1_off+i] = uf[k_off+jm_1_off+i]
											  -wm*(t[k_1_off+jmm1_1_off+i]
												  -t[k_A1_off+jmm1_1_off+i]);

						vf[k_off+jm_1_off+i] = vf[k_off+jm_1_off+i]
											  -wm*(s[k_1_off+jmm1_1_off+i]
												  -s[k_A1_off+jmm1_1_off+i]);
					}
				}
			}

			if (nfn > 3){
				for (k = 0; k < kbm1; k++){
					for (j = 0; j < nfn; j++){
						int jj = jm-j-1;	
						uf[k_off+jj*im+i] = uf[k_off+jj*im+i]
												*(1.f-frz[jj*im+i])
										   +(tobn[k_off+j_off+i]
												*frz[jj*im+i]);

						vf[k_off+jj*im+i] = vf[k_off+jj*im+i]
												*(1.f-frz[jj*im+i])
										   +(sobn[k_off+j_off+i]
												*frz[jj*im+i]);
					
					}
					//if (i < im && j >= jm-nfn && j < jm){
					//	int jj = jm-j-1;	
					//	uf[k_off+j_off+i] = uf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i])
					//					   +(tobn[k_off+jj*im+i]
					//							*frz[j_off+i]);

					//	vf[k_off+j_off+i] = vf[k_off+j_off+i]
					//							*(1.f-frz[j_off+i])
					//					   +(sobn[k_off+jj*im+i]
					//							*frz[j_off+i]);
					//}
				}
			}
		}
	}
}

__global__ void
bcond_overlap_ew_gpu_kernel_4_1(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ fsm,
				   int kb, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i, k;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	if (j < jm-1){
		for (k = 0; k < kb-1; k++){
			uf[k_off+j_off+i] *= fsm[j_off+i];
			vf[k_off+j_off+i] *= fsm[j_off+i];
		}
	}
	return;
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_4_1(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ fsm,
				   int n_east, int n_west, 
				   int kb, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int k;

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kb-1; k++){
				uf[k_off+j_off+im-1] *= fsm[j_off+im-1];
				vf[k_off+j_off+im-1] *= fsm[j_off+im-1];
			}
		}
	
	}

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kb-1; k++){
				uf[k_off+j_off] *= fsm[j_off];
				vf[k_off+j_off] *= fsm[j_off];
			}
		}
	}

	return;
}

__global__ void
bcond_overlap_sn_gpu_kernel_4_1(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ fsm,
				   int kb, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j, k;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){
		for (k = 0; k < kb-1; k++){
			uf[k_off+j_off+i] *= fsm[j_off+i];
			vf[k_off+j_off+i] *= fsm[j_off+i];
		}
	}

	return;
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_4_1(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ fsm,
				   int n_south, int n_north,
				   int kb, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int k;

	if (n_south == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb-1; k++){
				uf[k_off+i] *= fsm[i];
				vf[k_off+i] *= fsm[i];
			}
		}
	}
	if (n_north == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb-1; k++){
				uf[k_off+jm_1_off+i] *= fsm[jm_1_off+i];
				vf[k_off+jm_1_off+i] *= fsm[jm_1_off+i];
			}
		}
		
	}
	return;
}

__global__ void
bcond_overlap_ew_gpu_kernel_5(
				   float *w, float *fsm,
				   int kb, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i, k;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	//int jmm1 = jm-1; 
	//int imm1 = im-1; 
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				w[k][j][i] *= fsm[j][i];	
			}
		}
	}
	*/

	if (j < jm-1){
		for (k = 0; k < kb-1; k++){
			w[k_off+j_off+i] *= fsm[j_off+i];
		}
	}
	
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_5(
				   float *w, float *fsm,
				   int n_east, int n_west, 
				   int kb, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k;

	//int jmm1 = jm-1; 
	//int imm1 = im-1; 
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				w[k][j][i] *= fsm[j][i];	
			}
		}
	}
	*/

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kb-1; k++){
				w[k_off+j_off] *= fsm[j_off];
			}
		}
	}

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kb-1; k++){
				w[k_off+j_off+im-1] *= fsm[j_off+im-1];
			}
		}
	}
}

__global__ void
bcond_overlap_sn_gpu_kernel_5(
				   float *w, float *fsm,
				   int kb, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j, k;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	//int jmm1 = jm-1; 
	//int imm1 = im-1; 
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				w[k][j][i] *= fsm[j][i];	
			}
		}
	}
	*/

	if (i > 32 && i < im-33){
		for (k = 0; k < kb-1; k++){
			w[k_off+j_off+i] *= fsm[j_off+i];
		}
	}
	
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_5(
				   float *w, float *fsm,
				   int n_south, int n_north, 
				   int kb, int jm, int im){

	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	int k;

	//int jmm1 = jm-1; 
	//int imm1 = im-1; 
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				w[k][j][i] *= fsm[j][i];	
			}
		}
	}
	*/

	if (n_south == -1){
		if (i > 0 && i < im){
			for (k = 0; k < kb-1; k++){
				w[k_off+i] *= fsm[i];	
			}
		}
	}

	if (n_north== -1){
		if (i > 0 && i < im){
			for (k = 0; k < kb-1; k++){
				w[k_off+jm_1_off+i] *= fsm[jm_1_off+i];	
			}
		}
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_6_0(
					 float *u,
				     float *uf, float *vf,
				     float *q2, float *q2l,
				     float *dx,
				     float dti, float small,
				     int n_east, int n_west, 
				     int kb, int jm, int im){
	float u1;
	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 

	//int kbm1 = kb-1;
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	//modify uf vf

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kb; k++){
				u1 = 2.0f*u[k_off+j_off+(im-1)]*dti
						 /(dx[j_off+(im-1)]+dx[j_off+(imm1-1)]);
				if (u1 <= 0){
					uf[k_off+j_off+(im-1)] = q2[k_off+j_off+(im-1)]
											-u1*(small
												-q2[k_off+j_off+(im-1)]);

					vf[k_off+j_off+(im-1)] = q2l[k_off+j_off+(im-1)]
											-u1*(small
												-q2l[k_off+j_off+(im-1)]);
				}else{
					uf[k_off+j_off+(im-1)] = q2[k_off+j_off+(im-1)]
											-u1*(q2[k_off+j_off+(im-1)]
												-q2[k_off+j_off+(imm1-1)]);
					vf[k_off+j_off+(im-1)] = q2l[k_off+j_off+(im-1)]
											-u1*(q2l[k_off+j_off+(im-1)]
												-q2l[k_off+j_off+(imm1-1)]);
				}
			}
		}
	}

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kb; k++){
				u1 = 2.0f*u[k_off+j_off+1]
						 *dti/(dx[j_off]+dx[j_off+1]);
				if (u1 >= 0){
					uf[k_off+j_off] = q2[k_off+j_off]
									 -u1*(q2[k_off+j_off]-small);	
					vf[k_off+j_off] = q2l[k_off+j_off]
									 -u1*(q2l[k_off+j_off]-small);
				}else{
					uf[k_off+j_off] = q2[k_off+j_off]
									 -u1*(q2[k_off+j_off+1]
										 -q2[k_off+j_off]);	

					vf[k_off+j_off] = q2l[k_off+j_off]
									 -u1*(q2l[k_off+j_off+1]
										 -q2l[k_off+j_off]);
				}
			}
		}
	}
}

__global__ void
bcond_overlap_sn_bcond_gpu_kernel_6_0(
					 float *v, 
				     float *uf, float *vf,
				     float *q2, float *q2l,
				     float *dy, 
				     float dti, float small,
				     int n_south, int n_north,
				     int kb, int jm, int im){
	float u1;
	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 

	//int kbm1 = kb-1;
	int jmm1 = jm-1; 
	int imm1 = im-1; 

	//modify uf vf

	if (n_south == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb; k++){
				u1 = 2.0f*v[k_off+1*im+i]*dti/(dy[i]+dy[im+i]);
				if (u1 >= 0){
					uf[k_off+i] = q2[k_off+i]
								 -u1*(q2[k_off+i]-small);	
					vf[k_off+i] = q2l[k_off+i]
								 -u1*(q2l[k_off+i]-small);	
				}else{
					uf[k_off+i] = q2[k_off+i]
								 -u1*(q2[k_off+1*im+i]
									 -q2[k_off+i]);	
					vf[k_off+i] = q2l[k_off+i]
								 -u1*(q2l[k_off+1*im+i]
									 -q2l[k_off+i]);
				}
			}
		}
	}

	if (n_north == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb; k++){
				u1 = 2.0f*v[k_off+jm_1_off+i]*dti
						 /(dy[jm_1_off+i]+dy[jmm1_1_off+i]);
				if (u1 <= 0){
					uf[k_off+jm_1_off+i] = q2[k_off+jm_1_off+i]
										  -u1*(small
											  -q2[k_off+jm_1_off+i]);	
					vf[k_off+jm_1_off+i] = q2l[k_off+jm_1_off+i]
										  -u1*(small
											  -q2l[k_off+jm_1_off+i]);
				}else{
					uf[k_off+jm_1_off+i] = q2[k_off+jm_1_off+i]
										  -u1*(q2[k_off+jm_1_off+i]
											  -q2[k_off+jmm1_1_off+i]);

					vf[k_off+jm_1_off+i] = q2l[k_off+jm_1_off+i]
										  -u1*(q2l[k_off+jm_1_off+i]
											  -q2l[k_off+jmm1_1_off+i]);
				}
			}
		}
	}
}

__global__ void
bcond_overlap_ew_gpu_kernel_6_1(float *uf, float *vf,
				     float *fsm,
				     int kb, int jm, int im){
	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	if (j < jm-1){
		for (k = 0; k < kb; k++){
			uf[k_off+j_off+i] = uf[k_off+j_off+i]*fsm[j_off+i];
			vf[k_off+j_off+i] = vf[k_off+j_off+i]*fsm[j_off+i]; 
		}
	}
}


__global__ void
bcond_overlap_sn_gpu_kernel_6_1(float *uf, float *vf,
				     float *fsm,
				     int kb, int jm, int im){
	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){
		for (k = 0; k < kb; k++){
			uf[k_off+j_off+i] = uf[k_off+j_off+i]*fsm[j_off+i];
			vf[k_off+j_off+i] = vf[k_off+j_off+i]*fsm[j_off+i]; 
		}
	}
}

__global__ void
bcond_overlap_ew_bcond_gpu_kernel_6_1(float *uf, float *vf,
				     float *fsm,
					 int n_east, int n_west,
				     int kb, int jm, int im){
	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kb; k++){
				uf[k_off+j_off] = uf[k_off+j_off]*fsm[j_off];
				vf[k_off+j_off] = vf[k_off+j_off]*fsm[j_off]; 
			}
		}
	}

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kb; k++){
				uf[k_off+j_off+im-1] = uf[k_off+j_off+im-1]*fsm[j_off+im-1];
				vf[k_off+j_off+im-1] = vf[k_off+j_off+im-1]*fsm[j_off+im-1]; 
			}
		}
	}
}


__global__ void
bcond_overlap_sn_bcond_gpu_kernel_6_1(float *uf, float *vf,
				     float *fsm,
					 int n_south, int n_north,
				     int kb, int jm, int im){
	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 

	if (n_south == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb; k++){
				uf[k_off+i] = uf[k_off+i]*fsm[i];
				vf[k_off+i] = vf[k_off+i]*fsm[i]; 
			}
		}
	}

	if (n_north == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb; k++){
				uf[k_off+jm_1_off+i] = uf[k_off+jm_1_off+i]*fsm[jm_1_off+i];
				vf[k_off+jm_1_off+i] = vf[k_off+jm_1_off+i]*fsm[jm_1_off+i]; 
			}
		}
	}
}


void bcond_overlap(int idx, cudaStream_t &stream_in){

#ifndef TIME_DISABLE
	struct timeval start_bcond,
				   end_bcond;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond);
#endif

	/*
    int i,j,k;
    float ga,u1,wm;
    float hmax;
	*/

	//dim3 threadPerBlock(block_i_2D, block_j_2D);
	//dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
	//				  (j_size+block_j_2D-1)/block_j_2D);

	//dim3 threadPerBlock_ew(1, 128);
	//dim3 blockPerGrid_ew(1, (j_size+127)/128);
	//dim3 threadPerBlock_sn(128, 1);
	//dim3 blockPerGrid_sn((i_size+127)/128, 1);

	//dim3 threadPerBlock_ew_32(32, 4);
	//dim3 blockPerGrid_ew_32(2, (j_size-2+3)/4);
	//dim3 threadPerBlock_sn_32(32, 4);
	//dim3 blockPerGrid_sn_32((i_size-2+31)/32, 16);

	if (idx == 1){

#ifndef TIME_DISABLE
	struct timeval start_bcond_1,
				   end_bcond_1;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_1);
#endif
		bcond_overlap_ew_bcond_gpu_kernel_1_0<<<blockPerGrid_ew_b1, 
								        threadPerBlock_ew_b1,
										0, stream_in>>>(
				d_elf, n_west, n_east, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_1_0<<<blockPerGrid_sn_b1, 
								        threadPerBlock_sn_b1,
										0, stream_in>>>(
				d_elf, n_north, n_south, jm, im);


		checkCudaErrors(cudaStreamSynchronize(stream_in));

		if (iperx != 0){
			//xperi2d_cuda_ipc(d_elf, d_elf_east_most, d_elf_west_most,
			//				 stream_in, im, jm);
			xperi2d_cudaUVA(d_elf, 
							d_elf_east_most, d_elf_west_most,
							stream_in, im, jm);
		}
		if (ipery != 0){
			printf("yperi2d_ipc is not supported now! File:%s, func:%s\n",
					__FILE__, __func__);
			yperi2d_mpi_gpu(d_elf, im, jm);
		}

		bcond_overlap_ew_gpu_kernel_1_1<<<blockPerGrid_ew_32, 
								        threadPerBlock_ew_32,
										0, stream_in>>>(
				d_elf, d_fsm, jm, im);

		bcond_overlap_sn_gpu_kernel_1_1<<<blockPerGrid_sn_32, 
								        threadPerBlock_sn_32,
										0, stream_in>>>(
				d_elf, d_fsm, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_1_1<<<blockPerGrid_ew_b1, 
												threadPerBlock_ew_b1,
												0, stream_in>>>(
				d_elf, d_fsm, n_east, n_west, jm, im);


		bcond_overlap_sn_bcond_gpu_kernel_1_1<<<blockPerGrid_sn_b1, 
												threadPerBlock_sn_b1,
												0, stream_in>>>(
				d_elf, d_fsm, n_south, n_north, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));


#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_1);
		bcond_time_1 += time_consumed(&start_bcond_1, 
											  &end_bcond_1);
#endif
		//return;
	}
	else if (idx == 2){

#ifndef TIME_DISABLE
	struct timeval start_bcond_2,
				   end_bcond_2;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_2);
#endif

		bcond_overlap_ew_bcond_gpu_kernel_2_0<<<blockPerGrid_ew_b1, 
										  threadPerBlock_ew_b1,
										  0, stream_in>>>(
				d_uaf, d_vaf, d_el, d_uabe, d_uabw, 
				d_ele, d_elw, d_h, 
				rfe, rfw, ramp, grav, 
				n_east, n_west, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_2_0<<<blockPerGrid_sn_b1, 
										  threadPerBlock_sn_b1,
										  0, stream_in>>>(
				d_uaf, d_vaf, d_el, d_vabs, d_vabn, 
				d_els, d_eln, d_h, 
				rfs, rfn, ramp, grav, 
				n_south, n_north, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		if (iperx != 0){
			//xperi2d_cuda_ipc(d_uaf, d_uaf_east_most, d_uaf_west_most,
			//				 stream_in, im, jm);
			//xperi2d_cuda_ipc(d_vaf, d_vaf_east_most, d_vaf_west_most,
			//				 stream_in, im, jm);

			MPI_Barrier(pom_comm);

			xperi2d_cudaUVAAsync(d_uaf, 
								 d_uaf_east_most, d_uaf_west_most,
								 stream_in, im, jm);
			xperi2d_cudaUVAAsync(d_vaf, 
								 d_vaf_east_most, d_vaf_west_most,
							     stream_in, im, jm);
			
			checkCudaErrors(cudaStreamSynchronize(stream_in));
			MPI_Barrier(pom_comm);

			if (iperx < 0){
				bcond_overlap_sn_bcond_gpu_kernel_2_1<<<blockPerGrid_sn_b1, 
												  threadPerBlock_sn_b1,
												  0, stream_in>>>(
						d_uaf, d_dum, n_north, n_south, jm, im);
			}
		}

		if (ipery != 0){
			printf("yperi2d_cuda_ipc is not supported!File:%s, Func:%s\n",
					__FILE__, __func__);
			yperi2d_mpi_gpu(d_uaf, im, jm);
			yperi2d_mpi_gpu(d_vaf, im, jm);

			if (ipery < 0){
				bcond_overlap_ew_bcond_gpu_kernel_2_2<<<blockPerGrid_ew_b1, 
												  threadPerBlock_ew_b1,
												  0, stream_in>>>(
						d_vaf, d_dvm, n_east, n_west, jm, im);
			}
		}

		bcond_overlap_ew_gpu_kernel_2_3<<<blockPerGrid_ew_32, 
										  threadPerBlock_ew_32,
										  0, stream_in>>>(
				d_uaf, d_vaf, d_dum, d_dvm, 
				jm, im);

		bcond_overlap_sn_gpu_kernel_2_3<<<blockPerGrid_sn_32, 
										  threadPerBlock_sn_32,
										  0, stream_in>>>(
				d_uaf, d_vaf, d_dum, d_dvm, 
				jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_2_3<<<blockPerGrid_ew_b1, 
										  threadPerBlock_ew_b1,
										  0, stream_in>>>(
				d_uaf, d_vaf, d_dum, d_dvm, 
				n_east, n_west, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_2_3<<<blockPerGrid_sn_b1, 
										  threadPerBlock_sn_b1,
										  0, stream_in>>>(
				d_uaf, d_vaf, d_dum, d_dvm, 
				n_south, n_north, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_2);
		bcond_time_2 += time_consumed(&start_bcond_2, 
								      &end_bcond_2);
#endif
		//return;
	}
//	/*
//      else if(idx.eq.3) then
//
//! internal (3-D) velocity boundary conditions
//
//        do k=1,kbm1
//          do j=1,jm
//            do i=1,im
//              uf(i,j,k)=uf(i,j,k)*dum(i,j)
//              vf(i,j,k)=vf(i,j,k)*dvm(i,j)
//            end do
//          end do
//        end do
//
//        return
//
//	*/
//	
	else if(idx == 3){

#ifndef TIME_DISABLE
	struct timeval start_bcond_3,
				   end_bcond_3;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_3);
#endif
		/*
//!     EAST
//!     radiation boundary conditions.
		if (n_east == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					ga = sqrtf(h[j][im-1]/hmax);
					uf[k][j][im-1] = ga*(0.25f*u[k][j-1][imm1-1]
										+0.5f*u[k][j][imm1-1]
										+0.25f*u[k][j+1][imm1-1])
									+(1.f-ga)*(0.25f*u[k][j-1][im-1]
											  +0.5f*u[k][j][im-1]
											  +0.25f*u[k][j+1][im-1]);
					vf[k][j][im-1] = 0;
				}
			}
		}

//!     WEST
//!     radiation boundary conditions.
		if (n_west == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					ga = sqrtf(h[j][0]/hmax);
					uf[k][j][1] = ga*(0.25f*u[k][j-1][2]
									 +0.5f*u[k][j][2]
									 +0.25f*u[k][j+1][2])
								 +(1.f-ga)*(0.25f*u[k][j-1][1]
										   +0.5f*u[k][j][1]
										   +0.25f*u[k][j+1][1]);
					uf[k][j][0] = uf[k][j][1];
					vf[k][j][0] = 0;
				}
			}
		}

//!     NORTH
//!     radiation boundary conditions.
		
		if (n_north == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 1; i < imm1; i++){
					ga = sqrtf(h[jm-1][i]/hmax);
					vf[k][jm-1][i] = ga*(0.25f*v[k][jmm1-1][i-1]
										+0.5f*v[k][jmm1-1][i]
										+0.25f*v[k][jmm1-1][i+1])
									+(1.f-ga)*(0.25f*v[k][jm-1][i-1]
											  +0.5f*v[k][jm-1][i]
											  +0.25f*v[k][jm-1][i+1]);
					uf[k][jm-1][i] = 0;
				}
			}
		}

//!     SOUTH
//!     radiation boundary conditions.
		
		if (n_south == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 1; i < imm1; i++){
					ga = sqrtf(h[0][i]/hmax);	
					vf[k][1][i] = ga*(0.25f*v[k][2][i-1]
									 +0.5f*v[k][2][i]
									 +0.25*v[k][2][i+1])
								 +(1.f-ga)*(0.25f*v[k][1][i-1]
										   +0.5f*v[k][1][i]
										   +0.25f*v[k][1][i+1]);
					vf[k][0][i] = vf[k][1][i];
					uf[k][0][i] = 0;
				}
			}
		}

		if (iperx != 0){
			xperi2d_mpi(wubot, im, jm);
			xperi2d_mpi(wvbot, im, jm);
			xperi3d_mpi(uf, im, jm, kbm1);
			xperi3d_mpi(vf, im, jm, kbm1);

			if (iperx < 0){
				if (n_north == -1){
					for (i = 0; i < im; i++){
						wubot[jm-1][i] = wubot[jmm1-1][i];	
					}
					for (k = 0; k < kbm1; k++){
						for (i = 0; i < im; i++){
							uf[k][jm-1][i] = uf[k][jmm1-1][i];	
						}
					}
				}

				if (n_south == -1){
					for (i = 0; i < im; i++){
						wubot[0][i] = wubot[1][i];	
					}
					for (k = 0; k < kbm1; k++){
						for (i = 0; i < im; i++){
							uf[k][0][i] = uf[k][1][i];	
						}
					}
				}
				
			}

		}

		if (ipery != 0){
			yperi2d_mpi(wubot, im, jm);
			yperi2d_mpi(wvbot, im, jm);
			yperi3d_mpi(uf, im, jm, kbm1);
			yperi3d_mpi(vf, im, jm, kbm1);

			if (ipery < 0){
				if (n_east == -1){
					for (j = 0; j < jm; j++){
						wvbot[j][im-1] = wvbot[j][imm1-1];	
					}
					for (k = 0; k < kbm1; k++){
						for (j = 0; j < jm; j++){
							vf[k][j][im-1] = vf[k][j][imm1-1];	
						}
					}
				}

				if (n_west == -1){
					for (j = 0; j < jm; j++){
						wvbot[j][0] = wvbot[j][1];	
					}
					for (k = 0; k < kbm1; k++){
						for (j = 0; j < jm; j++){
							vf[k][j][0] = vf[k][j][1];	
						}
					}
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*dum[j][i];	
					vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
				}
			}
		}
		*/

		//bcond_gpu_kernel_3_0<<<blockPerGrid, threadPerBlock>>>(
		//		d_uf, d_vf, d_u, d_v, 
		//		d_h, n_east, n_west, n_south, n_north,
		//		kb, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_3_0<<<blockPerGrid_ew_b1, 
												threadPerBlock_ew_b1,
												0, stream_in>>>(
				d_uf, d_vf, d_u, 
				d_h, n_east, n_west, 
				kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_3_0<<<blockPerGrid_sn_b1, 
											    threadPerBlock_sn_b1,
												0, stream_in>>>(
				d_uf, d_vf, d_v, 
				d_h, n_south, n_north,
				kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		if (iperx != 0){

			//xperi2d_mpi_gpu(d_wubot, im, jm);
			//xperi2d_mpi_gpu(d_wvbot, im, jm);
			//xperi2d_cuda_aware_mpi(d_wubot, im, jm);
			//xperi2d_cuda_aware_mpi(d_wvbot, im, jm);
			//xperi2d_cuda_ipc(d_wubot, d_wubot_east_most, d_wubot_west_most, 
			//				 stream_in, im, jm);
			//xperi2d_cuda_ipc(d_wvbot, d_wvbot_east_most, d_wvbot_west_most,
			//				 stream_in, im, jm);

			//xperi3d_mpi_gpu(d_uf, im, jm, kbm1);
			//xperi3d_mpi_gpu(d_vf, im, jm, kbm1);
			//xperi3d_cuda_aware_mpi(d_uf, im, jm, kbm1);
			//xperi3d_cuda_aware_mpi(d_vf, im, jm, kbm1);

			//xperi3d_cuda_ipc(d_uf, d_uf_east_most, d_uf_west_most,
			//				 stream_in, im, jm, kbm1);
			//xperi3d_cuda_ipc(d_vf, d_vf_east_most, d_vf_west_most,
			//				 stream_in, im, jm, kbm1);

			MPI_Barrier(pom_comm);

			xperi2d_cudaUVAAsync(d_wubot, 
								 d_wubot_east_most, d_wubot_west_most, 
								 stream_in, im, jm);
			xperi2d_cudaUVAAsync(d_wvbot, 
								 d_wvbot_east_most, d_wvbot_west_most,
								 stream_in, im, jm);
			xperi3d_cudaUVAAsync(d_uf, 
								 d_uf_east_most, d_uf_west_most,
								 stream_in, im, jm, kbm1);
			xperi3d_cudaUVAAsync(d_vf, 
								 d_vf_east_most, d_vf_west_most,
								 stream_in, im, jm, kbm1);

			checkCudaErrors(cudaStreamSynchronize(stream_in));
			MPI_Barrier(pom_comm);

			if (iperx < 0){
				bcond_overlap_sn_bcond_gpu_kernel_3_1<<<blockPerGrid_sn_b1, 
													threadPerBlock_sn_b1,
													0, stream_in>>>(
						d_wubot, d_uf, d_vf, 
						n_south, n_north, 
						kb, jm, im);
				//if (n_north == -1){
				//	checkCudaErrors(cudaMemcpy(d_wubot+(jm-1)*im,
				//							   d_wubot+(jmm1-1)*im,
				//							   im*sizeof(float),
				//							   cudaMemcpyDeviceToDevice));
				//	checkCudaErrors(cudaMemcpy2D(d_uf+(jm-1)*im,
				//								jm*im*sizeof(float),
				//								d_uf+(jmm1-1)*im,
				//								jm*im*sizeof(float),
				//								im*sizeof(float),
				//								kbm1,
				//								cudaMemcpyDeviceToDevice));
				//	//for (i = 0; i < im; i++){
				//	//	wubot[jm-1][i] = wubot[jmm1-1][i];	
				//	//}
				//	//for (k = 0; k < kbm1; k++){
				//	//	for (i = 0; i < im; i++){
				//	//		uf[k][jm-1][i] = uf[k][jmm1-1][i];	
				//	//	}
				//	//}
				//}

				//checkCudaErrors(cudaDeviceSynchronize());

				//if (n_south == -1){
				//	checkCudaErrors(cudaMemcpy(d_wubot,
				//							   d_wubot+im,
				//							   im*sizeof(float),
				//							   cudaMemcpyDeviceToDevice));
				//	
				//	checkCudaErrors(cudaMemcpy2D(d_uf,
				//								jm*im*sizeof(float),
				//								d_uf+im,
				//								jm*im*sizeof(float),
				//								im*sizeof(float),
				//								kbm1,
				//								cudaMemcpyDeviceToDevice));
				//	//for (i = 0; i < im; i++){
				//	//	wubot[0][i] = wubot[1][i];	
				//	//}
				//	//for (k = 0; k < kbm1; k++){
				//	//	for (i = 0; i < im; i++){
				//	//		uf[k][0][i] = uf[k][1][i];	
				//	//	}
				//	//}
				//}
			}
		}


		if (ipery != 0){

			printf("ipery != 0 the feature is not supported!, FILE=%s, LINE:%d\n",
					__FILE__, __LINE__);
			yperi2d_mpi_gpu(d_wubot, im, jm);
			yperi2d_mpi_gpu(d_wvbot, im, jm);
			yperi3d_mpi_gpu(d_uf, im, jm, kbm1);
			yperi3d_mpi_gpu(d_vf, im, jm, kbm1);

			if (ipery < 0){
				if (n_east == -1){
					//for (j = 0; j < jm; j++){
					//	wvbot[j][im-1] = wvbot[j][imm1-1];	
					//}
					//for (k = 0; k < kbm1; k++){
					//	for (j = 0; j < jm; j++){
					//		vf[k][j][im-1] = vf[k][j][imm1-1];	
					//	}
					//}
					checkCudaErrors(cudaMemcpy2D(d_wvbot+im-1,
												 im*sizeof(float),
												 d_wvbot+imm1-1,
												 im*sizeof(float),
												 sizeof(float),
												 jm,
												 cudaMemcpyDeviceToDevice));

					checkCudaErrors(cudaMemcpy2D(d_vf+im-1,
												 im*sizeof(float),
												 d_vf+imm1-1,
												 im*sizeof(float),
												 sizeof(float),
												 kbm1*jm,
												 cudaMemcpyDeviceToDevice));

				}

				if (n_west == -1){
					//for (j = 0; j < jm; j++){
					//	wvbot[j][0] = wvbot[j][1];	
					//}
					//for (k = 0; k < kbm1; k++){
					//	for (j = 0; j < jm; j++){
					//		vf[k][j][0] = vf[k][j][1];	
					//	}
					//}
					checkCudaErrors(cudaMemcpy2D(d_wvbot,
												 im*sizeof(float),
												 d_wvbot+1,
												 im*sizeof(float),
												 sizeof(float),
												 jm,
												 cudaMemcpyDeviceToDevice));

					checkCudaErrors(cudaMemcpy2D(d_vf,
												 im*sizeof(float),
												 d_vf+1,
												 im*sizeof(float),
												 sizeof(float),
												 kbm1*jm,
												 cudaMemcpyDeviceToDevice));
				}
			}
		}


		//bcond_gpu_kernel_3_1<<<blockPerGrid, threadPerBlock>>>(
		//		d_uf, d_vf, d_dum, d_dvm,
		//		kb, jm, im);
		bcond_overlap_ew_gpu_kernel_3_2<<<blockPerGrid_ew_32, 
										  threadPerBlock_ew_32,
										  0, stream_in>>>(
				d_uf, d_vf, d_dum, d_dvm,
				kb, jm, im);

		bcond_overlap_sn_gpu_kernel_3_2<<<blockPerGrid_sn_32, 
										  threadPerBlock_sn_32,
										  0, stream_in>>>(
				d_uf, d_vf, d_dum, d_dvm,
				kb, jm, im);


		bcond_overlap_ew_bcond_gpu_kernel_3_2<<<blockPerGrid_ew_b1, 
										  threadPerBlock_ew_b1,
										  0, stream_in>>>(
				d_uf, d_vf, d_dum, d_dvm,
				n_east, n_west, kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_3_2<<<blockPerGrid_sn_b1, 
										  threadPerBlock_sn_b1,
										  0, stream_in>>>(
				d_uf, d_vf, d_dum, d_dvm,
				n_south, n_north, kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));


		/*
		checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

//		bcond_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
//				d_uf, d_vf, d_dum, d_dvm, kb, jm, im);

		/*
		bcond_gpu_kernel_3(float *uf, float *vf,
						   float *dum, float *dvm,
						   int kb, int jm, int im);
		*/
		/*
		checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_3);
		bcond_time_3 += time_consumed(&start_bcond_3, 
								      &end_bcond_3);
#endif
		//return;
	}
	
//	/*
//      else if(idx.eq.4) then
//
//! temperature and salinity boundary conditions (using uf and vf,
//! respectively)
//        do k=1,kbm1
//          do j=1,jm
//            ! east
//            if(n_east.eq.-1) then
//              u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
//              if(u1.le.0.e0) then
//                uf(im,j,k)=t(im,j,k)-u1*(tbe(j,k)-t(im,j,k))
//                vf(im,j,k)=s(im,j,k)-u1*(sbe(j,k)-s(im,j,k))
//              else
//                uf(im,j,k)=t(im,j,k)-u1*(t(im,j,k)-t(imm1,j,k))
//                vf(im,j,k)=s(im,j,k)-u1*(s(im,j,k)-s(imm1,j,k))
//                if(k.ne.1.and.k.ne.kbm1) then
//                  wm=.5e0*(w(imm1,j,k)+w(imm1,j,k+1))*dti
//     $                /((zz(k-1)-zz(k+1))*dt(imm1,j))
//                  uf(im,j,k)=uf(im,j,k)-wm*(t(imm1,j,k-1)-t(imm1,j,k+1))
//                  vf(im,j,k)=vf(im,j,k)-wm*(s(imm1,j,k-1)-s(imm1,j,k+1))
//                endif
//              end if
//            end if
//
//            ! west
//            if(n_west.eq.-1) then
//              u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
//              if(u1.ge.0.e0) then
//                uf(1,j,k)=t(1,j,k)-u1*(t(1,j,k)-tbw(j,k))
//                vf(1,j,k)=s(1,j,k)-u1*(s(1,j,k)-sbw(j,k))
//              else
//                uf(1,j,k)=t(1,j,k)-u1*(t(2,j,k)-t(1,j,k))
//                vf(1,j,k)=s(1,j,k)-u1*(s(2,j,k)-s(1,j,k))
//                if(k.ne.1.and.k.ne.kbm1) then
//                  wm=.5e0*(w(2,j,k)+w(2,j,k+1))*dti
//     $                /((zz(k-1)-zz(k+1))*dt(2,j))
//                  uf(1,j,k)=uf(1,j,k)-wm*(t(2,j,k-1)-t(2,j,k+1))
//                  vf(1,j,k)=vf(1,j,k)-wm*(s(2,j,k-1)-s(2,j,k+1))
//                end if
//              end if
//            end if
//          end do
//
//          do i=1,im
//            ! south
//            if(n_south.eq.-1) then
//              u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
//              if(u1.ge.0.e0) then
//                uf(i,1,k)=t(i,1,k)-u1*(t(i,1,k)-tbs(i,k))
//                vf(i,1,k)=s(i,1,k)-u1*(s(i,1,k)-sbs(i,k))
//              else
//                uf(i,1,k)=t(i,1,k)-u1*(t(i,2,k)-t(i,1,k))
//                vf(i,1,k)=s(i,1,k)-u1*(s(i,2,k)-s(i,1,k))
//                if(k.ne.1.and.k.ne.kbm1) then
//                  wm=.5e0*(w(i,2,k)+w(i,2,k+1))*dti
//     $                /((zz(k-1)-zz(k+1))*dt(i,2))
//                  uf(i,1,k)=uf(i,1,k)-wm*(t(i,2,k-1)-t(i,2,k+1))
//                  vf(i,1,k)=vf(i,1,k)-wm*(s(i,2,k-1)-s(i,2,k+1))
//                end if
//              end if
//            end if
//
//            ! north
//            if(n_north.eq.-1) then
//              u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
//              if(u1.le.0.e0) then
//                uf(i,jm,k)=t(i,jm,k)-u1*(tbn(i,k)-t(i,jm,k))
//                vf(i,jm,k)=s(i,jm,k)-u1*(sbn(i,k)-s(i,jm,k))
//              else
//                uf(i,jm,k)=t(i,jm,k)-u1*(t(i,jm,k)-t(i,jmm1,k))
//                vf(i,jm,k)=s(i,jm,k)-u1*(s(i,jm,k)-s(i,jmm1,k))
//                if(k.ne.1.and.k.ne.kbm1) then
//                  wm=.5e0*(w(i,jmm1,k)+w(i,jmm1,k+1))*dti
//     $                /((zz(k-1)-zz(k+1))*dt(i,jmm1))
//                  uf(i,jm,k)=uf(i,jm,k)-wm*(t(i,jmm1,k-1)-t(i,jmm1,k+1))
//                  vf(i,jm,k)=vf(i,jm,k)-wm*(s(i,jmm1,k-1)-s(i,jmm1,k+1))
//                end if
//              end if
//            end if
//          end do
//        end do
//
//        do k=1,kbm1
//          do j=1,jm
//            do i=1,im
//              uf(i,j,k)=uf(i,j,k)*fsm(i,j)
//              vf(i,j,k)=vf(i,j,k)*fsm(i,j)
//            end do
//          end do
//        end do
//
//        return
//
//	*/
//	
	else if (idx == 4){

#ifndef TIME_DISABLE
	struct timeval start_bcond_4,
				   end_bcond_4;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_4);
#endif

		//only modify uf & vf

		/*
		checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_t, t, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_s, s, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

		//bcond_gpu_kernel_4_0<<<blockPerGrid, threadPerBlock>>>(
		//		d_u, d_uf, d_v, d_vf, d_t, d_s, d_w, d_dt, 
		//		d_tobe, d_sobe, d_tobw, d_sobw,
		//		d_tobs, d_sobs, d_tobn, d_sobn,
		//		d_tbe, d_sbe, d_tbw, d_sbw,
		//		d_tbs, d_sbs, d_tbn, d_sbn,
		//		d_dx, d_dy, d_zz, d_frz,
		//		dti, n_east, n_west, n_south, n_north, kb, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_4_0<<<blockPerGrid_ew_b1, 
												threadPerBlock_ew_b1,
												0, stream_in>>>(
				d_u, d_uf, d_vf, d_t, d_s, d_w, d_dt, 
				d_tobe, d_sobe, d_tobw, d_sobw,
				d_tbe, d_sbe, d_tbw, d_sbw,
				d_dx, d_dy, d_zz, d_frz,
				dti, n_east, n_west, kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_4_0<<<blockPerGrid_sn_b1, 
												threadPerBlock_sn_b1,
												0, stream_in>>>(
				d_uf, d_v, d_vf, d_t, d_s, d_w, d_dt, 
				d_tobs, d_sobs, d_tobn, d_sobn,
				d_tbs, d_sbs, d_tbn, d_sbn,
				d_dx, d_dy, d_zz, d_frz,
				dti, n_south, n_north, kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		if (iperx != 0){
			//xperi3d_mpi_gpu(d_uf, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_vf, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_uf, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_vf, im, jm, kbm1);	

			//xperi3d_cuda_ipc(d_uf, d_uf_east, d_uf_west,
			//				 stream_in, im, jm, kbm1);	
			//xperi3d_cuda_ipc(d_vf, d_vf_east, d_vf_west,
			//				 stream_in, im, jm, kbm1);	

			MPI_Barrier(pom_comm);

			xperi3d_cudaUVAAsync(d_uf, d_uf_east, d_uf_west,
								 stream_in, im, jm, kbm1);	
			xperi3d_cudaUVAAsync(d_vf, d_vf_east, d_vf_west,
							     stream_in, im, jm, kbm1);	

			checkCudaErrors(cudaStreamSynchronize(stream_in));
			MPI_Barrier(pom_comm);
		}

		if (ipery != 0){
			printf("ipery != 0 is not supported! FILE:%s, LINE:%d\n",
					__FILE__, __LINE__);
			yperi3d_mpi_gpu(d_uf, im, jm, kbm1);	
			yperi3d_mpi_gpu(d_vf, im, jm, kbm1);	
		}

		//bcond_gpu_kernel_4_1<<<blockPerGrid, threadPerBlock>>>(
		//		d_uf, d_vf, d_fsm,
		//		kb, jm, im);

		bcond_overlap_ew_gpu_kernel_4_1<<<blockPerGrid_ew_32, 
										  threadPerBlock_ew_32,
										  0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				kb, jm, im);

		bcond_overlap_sn_gpu_kernel_4_1<<<blockPerGrid_sn_32, 
										  threadPerBlock_sn_32,
										  0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				kb, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_4_1<<<blockPerGrid_ew_b1, 
										        threadPerBlock_ew_b1,
												0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				n_east, n_west, kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_4_1<<<blockPerGrid_sn_b1, 
										        threadPerBlock_sn_b1,
												0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				n_south, n_north, kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));


		/*
		bcond_gpu_kernel_4(float *u, float *uf,
				   float *v, float *vf, 
				   float *t, float *s, 
				   float *w, float *dt, 
				   float *dx, float *dy,
				   float *tbe, float *sbe, 
				   float *tbw, float *sbw,
				   float *tbs, float *sbs,
				   float *tbn, float *sbn,
				   float *zz, float *fsm,
				   float dti, 
				   int n_east, int n_west, 
				   int n_south, int n_north,
				   int kb, int jm, int im){
		*/

		/*
		checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_4);
		bcond_time_4 += time_consumed(&start_bcond_4, 
								      &end_bcond_4);
#endif

		//return;
	}

//	/*
//      else if(idx.eq.5) then
//
//! vertical velocity boundary conditions
//        do k=1,kbm1
//          do j=1,jm
//            do i=1,im
//              w(i,j,k)=w(i,j,k)*fsm(i,j)
//            end do
//          end do
//        end do
//
//        return
//	*/
//	
	else if (idx == 5){

#ifndef TIME_DISABLE
	struct timeval start_bcond_5,
				   end_bcond_5;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_5);
#endif
		/*
		checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/
		if (iperx != 0){
			//xperi3d_mpi_gpu(d_w, im, jm, kbm1);
			//xperi3d_cuda_aware_mpi(d_w, im, jm, kbm1);
			//xperi3d_cuda_ipc(d_w, d_w_east_most, d_w_west_most,
			//				 stream_in, im, jm, kbm1);
			xperi3d_cudaUVA(d_w, d_w_east_most, d_w_west_most,
							stream_in, im, jm, kbm1);
		}

		if (ipery != 0){
			printf("ipery!=0, feature is not supported!File:%s, Line:%d\n",
					__FILE__, __LINE__);
			yperi3d_mpi_gpu(d_w, im, jm, kbm1);	
		}

		//bcond_gpu_kernel_5<<<blockPerGrid, threadPerBlock>>>(
		//		d_w, d_fsm, kb, jm, im);

		bcond_overlap_ew_gpu_kernel_5<<<blockPerGrid_ew_32, 
										threadPerBlock_ew_32,
										0, stream_in>>>(
				d_w, d_fsm, kb, jm, im);

		bcond_overlap_sn_gpu_kernel_5<<<blockPerGrid_sn_32, 
										threadPerBlock_sn_32,
										0, stream_in>>>(
				d_w, d_fsm, kb, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_5<<<blockPerGrid_ew_b1, 
											  threadPerBlock_ew_b1,
											  0, stream_in>>>(
				d_w, d_fsm, n_east, n_west, kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_5<<<blockPerGrid_sn_b1, 
										threadPerBlock_sn_b1,
										0, stream_in>>>(
				d_w, d_fsm, n_south, n_north, kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_5);
		bcond_time_5 += time_consumed(&start_bcond_5, 
								      &end_bcond_5);
#endif
		//return;
	}

//
//	/*
//      else if(idx.eq.6) then
//
//! q2 and q2l boundary conditions
//
//        do k=1,kb
//          do j=1,jm
//            ! west
//            if(n_west.eq.-1) then
//              u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
//              if(u1.ge.0.e0) then
//                uf(1,j,k)=q2(1,j,k)-u1*(q2(1,j,k)-small)
//                vf(1,j,k)=q2l(1,j,k)-u1*(q2l(1,j,k)-small)
//              else
//                uf(1,j,k)=q2(1,j,k)-u1*(q2(2,j,k)-q2(1,j,k))
//                vf(1,j,k)=q2l(1,j,k)-u1*(q2l(2,j,k)-q2l(1,j,k))
//              end if
//            end if
//
//            ! east
//            if(n_east.eq.-1) then
//              u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
//              if(u1.le.0.e0) then
//                uf(im,j,k)=q2(im,j,k)-u1*(small-q2(im,j,k))
//                vf(im,j,k)=q2l(im,j,k)-u1*(small-q2l(im,j,k))
//              else
//                uf(im,j,k)=q2(im,j,k)-u1*(q2(im,j,k)-q2(imm1,j,k))
//                vf(im,j,k)=q2l(im,j,k)-u1*(q2l(im,j,k)-q2l(imm1,j,k))
//              end if
//            end if
//          end do
//
//          do i=1,im
//            ! south
//            if(n_south.eq.-1) then
//              u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
//              if(u1.ge.0.e0) then
//                uf(i,1,k)=q2(i,1,k)-u1*(q2(i,1,k)-small)
//                vf(i,1,k)=q2l(i,1,k)-u1*(q2l(i,1,k)-small)
//              else
//                uf(i,1,k)=q2(i,1,k)-u1*(q2(i,2,k)-q2(i,1,k))
//                vf(i,1,k)=q2l(i,1,k)-u1*(q2l(i,2,k)-q2l(i,1,k))
//              end if
//            end if
//
//            ! north
//            if(n_north.eq.-1) then
//              u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
//              if(u1.le.0.e0) then
//                uf(i,jm,k)=q2(i,jm,k)-u1*(small-q2(i,jm,k))
//                vf(i,jm,k)=q2l(i,jm,k)-u1*(small-q2l(i,jm,k))
//              else
//                uf(i,jm,k)=q2(i,jm,k)-u1*(q2(i,jm,k)-q2(i,jmm1,k))
//                vf(i,jm,k)=q2l(i,jm,k)-u1*(q2l(i,jm,k)-q2l(i,jmm1,k))
//              end if
//            end if
//          end do
//        end do
//
//        do k=1,kb
//          do j=1,jm
//            do i=1,im
//              uf(i,j,k)=uf(i,j,k)*fsm(i,j)+1.e-10
//              vf(i,j,k)=vf(i,j,k)*fsm(i,j)+1.e-10
//            end do
//          end do
//        end do
//
//        return
//
//      end if
//	  
//	*/
//	
	else if (idx == 6){

#ifndef TIME_DISABLE
	struct timeval start_bcond_6,
				   end_bcond_6;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_bcond_6);
#endif
		//modify uf vf
		/*
		if (n_east == -1){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][im-1]*dti
						/(dx[j][im-1]+dx[j][imm1-1]);	
					if (u1 <= 0.0f){
						uf[k][j][im-1] = q2[k][j][im-1]-u1*(small-q2[k][j][im-1]);
						vf[k][j][im-1] = q2l[k][j][im-1]-u1*(small-q2l[k][j][im-1]);
					}else{
						uf[k][j][im-1] = q2[k][j][im-1]
							         -u1*(q2[k][j][im-1]-q2[k][j][imm1-1]);	

						vf[k][j][im-1] = q2l[k][j][im-1]
							         -u1*(q2l[k][j][im-1]-q2l[k][j][imm1-1]);
					}
				}
			}
		}

		if (n_west == -1){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][1]*dti/(dx[j][0]+dx[j][1]);	
					if (u1 >= 0.0f){
						uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][0]-small);
						vf[k][j][0] = q2l[k][j][0]-u1*(q2l[k][j][0]-small);
					}else{
						uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][1]-q2[k][j][0]);	
						vf[k][j][0] = q2l[k][j][0] - u1*(q2l[k][j][1]-q2l[k][j][0]);
					}
				}
			}
		}

		if (n_north == -1){
			for (k = 0; k < kb; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][jm-1][i]*dti/(dy[jm-1][i]+dy[jmm1-1][i]);		
					if (u1 <= 0.0f){
						uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(small-q2[k][jm-1][i]);	
						vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(small-q2l[k][jm-1][i]);
					}else{
						uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(q2[k][jm-1][i]-q2[k][jmm1-1][i]);	
						vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(q2l[k][jm-1][i]-q2l[k][jmm1-1][i]);
					}

				}
			}
		}

		if (n_south == -1){
			for (k = 0; k < kb; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][1][i]*dti/(dy[0][i]+dy[1][i]);	
					if (u1 >= 0.0f){
						uf[k][0][i] = q2[k][0][i]-u1*(q2[k][0][i]-small);	
						vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][0][i]-small);
					}else{
						uf[k][0][i] = q2[k][0][i]-u1*(q2[k][1][i]-q2[k][0][i]);	
						vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][1][i]-q2l[k][0][i]);
					}
				}
			}
		}
		*/
		//bcond_gpu_kernel_6_0<<<blockPerGrid, threadPerBlock>>>(
		//		d_u, d_v, d_uf, d_vf, d_q2, d_q2l, 
		//		d_dx, d_dy, 
		//		dti, small, n_east, n_west, n_south, n_north, 
		//		kb, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_6_0<<<blockPerGrid_ew_b1, 
										threadPerBlock_ew_b1,
										0, stream_in>>>(
				d_u, d_uf, d_vf, d_q2, d_q2l, 
				d_dx, 
				dti, small, n_east, n_west, 
				kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_6_0<<<blockPerGrid_sn_b1, 
										threadPerBlock_sn_b1,
										0, stream_in>>>(
				d_v, d_uf, d_vf, d_q2, d_q2l, 
				d_dy, 
				dti, small, n_south, n_north, 
				kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));
		
		if (iperx != 0){
			//xperi3d_mpi_gpu(d_uf, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_vf, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_kh, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_km, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_kq, im, jm, kbm1);	
			//xperi3d_mpi_gpu(d_l, im, jm, kbm1);	

			//xperi3d_cuda_aware_mpi(d_uf, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_vf, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_kh, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_km, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_kq, im, jm, kbm1);	
			//xperi3d_cuda_aware_mpi(d_l, im, jm, kbm1);	

			//xperi3d_cuda_ipc(d_uf, d_uf_east_most, d_uf_west_most,
			//				 stream_in, im, jm, kbm1);	
			//xperi3d_cuda_ipc(d_vf, d_vf_east_most, d_vf_west_most,
			//				 stream_in, im, jm, kbm1);	
			//xperi3d_cuda_ipc(d_kh, d_kh_east_most, d_kh_west_most,
			//				 stream_in, im, jm, kbm1);	
			//xperi3d_cuda_ipc(d_km, d_km_east_most, d_km_west_most,
			//				 stream_in, im, jm, kbm1);	
			//xperi3d_cuda_ipc(d_kq, d_kq_east_most, d_kq_west_most,
			//				 stream_in, im, jm, kbm1);	
			////xperi3d_cuda_ipc(d_l, im, jm, kbm1);	

			MPI_Barrier(pom_comm);

			xperi3d_cudaUVAAsync(d_uf, d_uf_east_most, d_uf_west_most,
								 stream_in, im, jm, kbm1);	
			xperi3d_cudaUVAAsync(d_vf, d_vf_east_most, d_vf_west_most,
								 stream_in, im, jm, kbm1);	
			xperi3d_cudaUVAAsync(d_kh, d_kh_east_most, d_kh_west_most,
								 stream_in, im, jm, kbm1);	
			xperi3d_cudaUVAAsync(d_km, d_km_east_most, d_km_west_most,
								 stream_in, im, jm, kbm1);	
			xperi3d_cudaUVAAsync(d_kq, d_kq_east_most, d_kq_west_most,
								 stream_in, im, jm, kbm1);	

			checkCudaErrors(cudaStreamSynchronize(stream_in));
			MPI_Barrier(pom_comm);
		}

		if (ipery != 0){
			printf("ipery != 0, this feature is not supported for cuda_ipc"
				   "File:%s, Line:%d\n", __FILE__, __LINE__);
			yperi3d_mpi_gpu(d_uf, im, jm, kbm1);
			yperi3d_mpi_gpu(d_vf, im, jm, kbm1);
			yperi3d_mpi_gpu(d_kh, im, jm, kbm1);
			yperi3d_mpi_gpu(d_km, im, jm, kbm1);
			yperi3d_mpi_gpu(d_kq, im, jm, kbm1);
			yperi3d_mpi_gpu(d_l, im, jm, kbm1);
		}

		/*
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*fsm[j][i];
					vf[k][j][i] = vf[k][j][i]*fsm[j][i];
				}
			}
		}
		*/

		//bcond_gpu_kernel_6_1<<<blockPerGrid, threadPerBlock>>>(
		//		d_uf, d_vf, d_fsm,
		//		kb, jm, im);

		bcond_overlap_ew_gpu_kernel_6_1<<<blockPerGrid_ew_32, 
							      threadPerBlock_ew_32,
								  0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				kb, jm, im);

		bcond_overlap_sn_gpu_kernel_6_1<<<blockPerGrid_sn_32, 
								  threadPerBlock_sn_32,
								  0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				kb, jm, im);

		bcond_overlap_ew_bcond_gpu_kernel_6_1<<<blockPerGrid_ew_b1, 
										threadPerBlock_ew_b1,
										0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				n_east, n_west, kb, jm, im);

		bcond_overlap_sn_bcond_gpu_kernel_6_1<<<blockPerGrid_sn_b1, 
									    threadPerBlock_sn_b1,
										0, stream_in>>>(
				d_uf, d_vf, d_fsm,
				n_south, n_north, kb, jm, im);

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		/*
		checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_q2, q2, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_q2l, q2l, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		*/

//		bcond_gpu_kernel_6<<<blockPerGrid, threadPerBlock>>>(
//				d_u, d_v, d_uf, d_vf, d_q2, d_q2l, d_dx, d_dy, d_fsm,
//				dti, small, n_east, n_west, n_south, n_north, kb, jm, im);
		/*
		bcond_gpu_kernel_6(float *u, float *v, 
				   float *uf, float *vf,
				   float *q2, float *q2l,
				   float *dx, float *dy,
				   float dti, float small,
				   int n_east, int n_west, 
				   int n_south, int n_north,
				   int kb, int jm, int im){
		*/

		/*
		checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		*/

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_bcond_6);
		bcond_time_6 += time_consumed(&start_bcond_6, 
								      &end_bcond_6);
#endif
		//return;
	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&end_bcond);
	bcond_time += time_consumed(&start_bcond, 
							    &end_bcond);
#endif
	
}


