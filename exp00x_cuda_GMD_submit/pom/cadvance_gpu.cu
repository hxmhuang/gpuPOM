#include<stdio.h>

#include"cadvance_gpu.h"
#include"cadvance_gpu_kernel.h"

#include"utils.h"
#include"cu_data.h"
#include"csolver_gpu.h"
#include"cbounds_forcing_gpu.h"
#include"cparallel_mpi_gpu.h"

#include"data.h"
#include"timer_all.h"
#include"cparallel_mpi.h"

#define NINT(i) ((int)(i+0.5f))

void get_time_gpu(){
	
	model_time = dti*((float)iint)/86400.f+time0;
	ramp = 1.0f;	


	return;
}


__global__ void
surface_forcing_gpu_kernel_0(float * __restrict__ e_atmos, 
							 float * __restrict__ swrad,
						     const float * __restrict__ vfluxf, 
							 float * __restrict__ w,
						     int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	//int jmm1 = jm-1;
	//int imm1 = im-1;

    //float tatm, satm;

	if (j < jm && i < im){

		e_atmos[j_off+i] = 0;

		w[j_off+i] = vfluxf[j_off+i];

		swrad[j_off+i] = 0;

		//tatm = t[j_off+i] + tbias;

		//satm = 0;
	}
}


void surface_forcing_gpu(){

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	/*
	checkCudaErrors(cudaMemcpy(d_vfluxf, vfluxf, jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	*/

	/*
	surface_forcing_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_wusurf, d_wvsurf, d_e_atmos, d_swrad, 
			d_vfluxf, d_w, d_wtsurf, d_wssurf, d_t, d_s, 
		    tbias, sbias, jm, im);
	*/

	surface_forcing_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_e_atmos, d_swrad, d_vfluxf, d_w, 
			jm, im);
	
	/*
	checkCudaErrors(cudaMemcpy(e_atmos, d_e_atmos, jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(swrad, d_swrad, jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(w, d_w, jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	*/
	
	
	//checkCudaErrors(cudaDeviceSynchronize());
    return;
}

__global__ void
momentum3d_gpu_kernel_0(float * __restrict__ aam, 
						float aam_init,
						int kb, int jm, int im){
	int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	for (k = 0; k < kb; k++){
		if (j < jm && i < im)
			aam[k_off+j_off+i] = aam_init;
	}

}

__global__ void
momentum3d_gpu_kernel_1(float * __restrict__ aam, 
						const float * __restrict__ aamfrz,
					    const float * __restrict__ u, 
						const float * __restrict__ v,
						const float * __restrict__ dx, 
						const float * __restrict__ dy,
						float horcon,
						int kb, int jm, int im){
	int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	if (j > 0 && j < jmm1 && 
		i > 0 && i < imm1){

		for (k = 0; k < kbm1; k++){

		    float tmp_u = (u[k_off+j_off+i+1]-u[k_off+j_off+i])
						  /dx[j_off+i];

		    float tmp_v = (v[k_off+j_A1_off+i]-v[k_off+j_off+i])
						  /dy[j_off+i];

		    float tmp_uv = (0.25f*(u[k_off+j_A1_off+i]
		    					  +u[k_off+j_A1_off+i+1]
		    					  -u[k_off+j_1_off+i]
		    					  -u[k_off+j_1_off+i+1])/dy[j_off+i]
		    			   +0.25f*(v[k_off+j_off+i+1]
		    			   	      +v[k_off+j_A1_off+i+1]
		    			   	      -v[k_off+j_off+i-1]
		    			   	      -v[k_off+j_A1_off+i-1])/dx[j_off+i]);

		    aam[k_off+j_off+i] = horcon*dx[j_off+i]*dy[j_off+i]
		    			*(1.f+aamfrz[j_off+i])		//!lyo:channel:
		    			*sqrtf((tmp_u*tmp_u)
		    				  +(tmp_v*tmp_v)
		    				  +0.5f*(tmp_uv*tmp_uv));
			}
	}

}

__global__ void
momentum3d_inner_gpu_kernel_1(float * __restrict__ aam, 
						const float * __restrict__ aamfrz,
					    const float * __restrict__ u, 
						const float * __restrict__ v,
						const float * __restrict__ dx, 
						const float * __restrict__ dy,
						float horcon,
						int kb, int jm, int im){
	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

	int kbm1 = kb-1;

	if (j > 32 && i > 32 && j < jm-33 && i < im-33){
	//if (j < jm-1 && i > 32 && i < im-33){

		for (k = 0; k < kbm1; k++){

		    float tmp_u = (u[k_off+j_off+i+1]-u[k_off+j_off+i])
						  /dx[j_off+i];

		    float tmp_v = (v[k_off+j_A1_off+i]-v[k_off+j_off+i])
						  /dy[j_off+i];

		    float tmp_uv = (0.25f*(u[k_off+j_A1_off+i]
		    					  +u[k_off+j_A1_off+i+1]
		    					  -u[k_off+j_1_off+i]
		    					  -u[k_off+j_1_off+i+1])/dy[j_off+i]
		    			   +0.25f*(v[k_off+j_off+i+1]
		    			   	      +v[k_off+j_A1_off+i+1]
		    			   	      -v[k_off+j_off+i-1]
		    			   	      -v[k_off+j_A1_off+i-1])/dx[j_off+i]);

		    aam[k_off+j_off+i] = horcon*dx[j_off+i]*dy[j_off+i]
		    			*(1.f+aamfrz[j_off+i])		//!lyo:channel:
		    			*sqrtf((tmp_u*tmp_u)
		    				  +(tmp_v*tmp_v)
		    				  +0.5f*(tmp_uv*tmp_uv));
		}
	}
}

__global__ void
momentum3d_ew_gpu_kernel_1(float * __restrict__ aam, 
						const float * __restrict__ aamfrz,
					    const float * __restrict__ u, 
						const float * __restrict__ v,
						const float * __restrict__ dx, 
						const float * __restrict__ dy,
						float horcon,
						int kb, int jm, int im){
	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	int kbm1 = kb-1;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	if (j < jm-1){

		for (k = 0; k < kbm1; k++){

		    float tmp_u = (u[k_off+j_off+i+1]-u[k_off+j_off+i])
						  /dx[j_off+i];

		    float tmp_v = (v[k_off+j_A1_off+i]-v[k_off+j_off+i])
						  /dy[j_off+i];

		    float tmp_uv = (0.25f*(u[k_off+j_A1_off+i]
		    					  +u[k_off+j_A1_off+i+1]
		    					  -u[k_off+j_1_off+i]
		    					  -u[k_off+j_1_off+i+1])/dy[j_off+i]
		    			   +0.25f*(v[k_off+j_off+i+1]
		    			   	      +v[k_off+j_A1_off+i+1]
		    			   	      -v[k_off+j_off+i-1]
		    			   	      -v[k_off+j_A1_off+i-1])/dx[j_off+i]);

		    aam[k_off+j_off+i] = horcon*dx[j_off+i]*dy[j_off+i]
		    			*(1.f+aamfrz[j_off+i])		//!lyo:channel:
		    			*sqrtf((tmp_u*tmp_u)
		    				  +(tmp_v*tmp_v)
		    				  +0.5f*(tmp_uv*tmp_uv));
			}
	}

}

__global__ void
momentum3d_sn_gpu_kernel_1(float * __restrict__ aam, 
						const float * __restrict__ aamfrz,
					    const float * __restrict__ u, 
						const float * __restrict__ v,
						const float * __restrict__ dx, 
						const float * __restrict__ dy,
						float horcon,
						int kb, int jm, int im){
	int k;

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	int kbm1 = kb-1;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){ 

		for (k = 0; k < kbm1; k++){

		    float tmp_u = (u[k_off+j_off+i+1]-u[k_off+j_off+i])
						  /dx[j_off+i];

		    float tmp_v = (v[k_off+j_A1_off+i]-v[k_off+j_off+i])
						  /dy[j_off+i];

		    float tmp_uv = (0.25f*(u[k_off+j_A1_off+i]
		    					  +u[k_off+j_A1_off+i+1]
		    					  -u[k_off+j_1_off+i]
		    					  -u[k_off+j_1_off+i+1])/dy[j_off+i]
		    			   +0.25f*(v[k_off+j_off+i+1]
		    			   	      +v[k_off+j_A1_off+i+1]
		    			   	      -v[k_off+j_off+i-1]
		    			   	      -v[k_off+j_A1_off+i-1])/dx[j_off+i]);

		    aam[k_off+j_off+i] = horcon*dx[j_off+i]*dy[j_off+i]
		    			*(1.f+aamfrz[j_off+i])		//!lyo:channel:
		    			*sqrtf((tmp_u*tmp_u)
		    				  +(tmp_v*tmp_v)
		    				  +0.5f*(tmp_uv*tmp_uv));
			}
	}

}

/*
void momentum3d_c_(float f_advx[][j_size][i_size],
				float f_advy[][j_size][i_size],
				float f_drhox[][j_size][i_size],
				float f_drhoy[][j_size][i_size],
				float f_aam[][j_size][i_size],

				float f_u[][j_size][i_size],
				float f_v[][j_size][i_size],
				float f_ub[][j_size][i_size],
				float f_vb[][j_size][i_size],
				float f_rho[][j_size][i_size],
				float f_rmean[][j_size][i_size],
				float f_dt[][i_size],
				float f_dum[][i_size],
				float f_dvm[][i_size],
				float f_d[][i_size]){	//!lyo:!stokes:change subr name
*/

void momentum3d_gpu(){
//! formerly subroutine lateral_viscosity
//! calculate horizontal 3d-Momentum terms including the lateral viscosity
	//int i, j, k;

#ifndef TIME_DISABLE
	struct timeval start_momentum3d,
				   end_momentum3d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_momentum3d);

#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	dim3 threadPerBlock_ew(32, 4);
	dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	dim3 threadPerBlock_sn(32, 4);
	dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	if (mode != 2){
//------------------------------------------------------------------------
		/*
		checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_ub, ub, kb*jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vb, vb, kb*jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_rho, rho, kb*jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(d_rmean, rmean, kb*jm*im*sizeof(float), 
		//					       cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(d_dum, dum, jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_dvm, dvm, jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_d, d, jm*im*sizeof(float), 
							       cudaMemcpyHostToDevice));
		*/
		/*
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					u[k][j][i] = f_u[k][j][i];
					v[k][j][i] = f_v[k][j][i];
					ub[k][j][i] = f_ub[k][j][i];
					vb[k][j][i] = f_vb[k][j][i];
					aam[k][j][i] = f_aam[k][j][i];
					//advx[k][j][i] = f_advx[k][j][i];
					//advy[k][j][i] = f_advy[k][j][i];
				}
			}
		}

		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					rho[k][j][i] = f_rho[k][j][i];
					rmean[k][j][i] = f_rmean[k][j][i];
				}
			}
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				dum[j][i] = f_dum[j][i];
				dvm[j][i] = f_dvm[j][i];
				dt[j][i] = f_dt[j][i];
				d[j][i] = f_d[j][i];
			}
		}
		*/

		
		//advct(a, c, ee);
		//advct(advx,v,u,dt,ub,aam,vb,advy);
		advct_gpu();

//------------------------------------------------------------------------
		if (calc_stokes){
			printf("calc_stokes = TRUE!\n");
			printf("But stokes function is not implemented now...\n");
			exit(1);
			//stokes(ee);		//!(a,c,ee) !lyo:!stokes:
		}
//------------------------------------------------------------------------




		if (npg == 1){
			//baropg();	
			//baropg(rho, rmean, dum, dvm, dt, drhox, drhoy, ramp);
			baropg_gpu();
		}else if (npg == 2){
			//baropg_mcc();	
			//baropg_mcc(rho, rmean, d, dum, dvm, dt, drhox, drhoy, ramp);	
			printf("npg == 2!! but GPU_version does not supported now\n");
			exit(1);

		}else{
			error_status = 1;	
			printf("Error: invalid value for npg, File:%s, Func:%s, Line:%d",
					__FILE__, __func__, __LINE__);
		}

//------------------------------------------------------------------------

//! if mode=2 then initial values of aam2d are used. If one wishes
//! to use Smagorinsky lateral viscosity and diffusion for an
//! external (2-D) mode calculation, then appropiate code can be
//! adapted from that below and installed just before the end of the
//! "if(mode.eq.2)" loop in subroutine advave
//
//! calculate Smagorinsky lateral viscosity:
//! ( hor visc = horcon*dx*dy*sqrt((du/dx)**2+(dv/dy)**2
//!                                +.5*(du/dy+dv/dx)**2) )
//!lyo:scs1d:
		
		if (n1d != 0){
			/*
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						aam[k][j][i] = aam_init;	
					}
				}
			}
			*/
			momentum3d_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
				d_aam, aam_init,
				kb, jm, im);

		}else{
			/*
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					for (i = 1; i < imm1; i++){
						float tmp_u = (u[k][j][i+1]-u[k][j][i])/dx[j][i];
						float tmp_v = (v[k][j+1][i]-v[k][j][i])/dy[j][i];
						float tmp_uv = (0.25f*(u[k][j+1][i]
											  +u[k][j+1][i+1]
											  -u[k][j-1][i]
											  -u[k][j-1][i+1])/dy[j][i]
									   +0.25f*(v[k][j][i+1]
									   	      +v[k][j+1][i+1]
									   	      -v[k][j][i-1]
									   	      -v[k][j+1][i-1])/dx[j][i]);

						aam[k][j][i]=horcon*dx[j][i]*dy[j][i]
									*(1.f+aamfrz[j][i])		//!lyo:channel:
									*sqrtf((tmp_u*tmp_u)
										  +(tmp_v*tmp_v)
										  +0.5f*(tmp_uv*tmp_uv));
					}
				}
			}
			*/

			//momentum3d_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			//	d_aam, d_aamfrz, d_u, d_v, d_dx, d_dy,
			//	horcon, kb, jm, im);

			//exchange3d_cuda_aware_mpi(d_aam, im, jm, kbm1);

			momentum3d_ew_gpu_kernel_1<<<blockPerGrid_ew, threadPerBlock_ew,
										 0, stream[1]>>>(
				d_aam, d_aamfrz, d_u, d_v, d_dx, d_dy,
				horcon, kb, jm, im);

			momentum3d_sn_gpu_kernel_1<<<blockPerGrid_sn, threadPerBlock_sn,
										 0, stream[2]>>>(
				d_aam, d_aamfrz, d_u, d_v, d_dx, d_dy,
				horcon, kb, jm, im);

			momentum3d_inner_gpu_kernel_1<<<blockPerGrid, threadPerBlock,
											0, stream[0]>>>(
				d_aam, d_aamfrz, d_u, d_v, d_dx, d_dy,
				horcon, kb, jm, im);

			checkCudaErrors(cudaStreamSynchronize(stream[1]));
			checkCudaErrors(cudaStreamSynchronize(stream[2]));

			//exchange3d_mpi_gpu(d_aam, im, jm, kbm1);
			exchange3d_cudaUVA(d_aam, d_aam_east, d_aam_west,
							   d_aam_south, d_aam_north,
							   stream[1], im, jm, kbm1);

			//MPI_Barrier(pom_comm);
			//exchange3d_cuda_ipc(d_aam, d_aam_east, d_aam_west,
			//					stream[1], im, jm, kbm1);

			//checkCudaErrors(cudaStreamSynchronize(stream[1]));
			//MPI_Barrier(pom_comm);
			checkCudaErrors(cudaStreamSynchronize(stream[0]));

		}

		//exchange3d_mpi_gpu(d_aam, im, jm, kbm1);
		//exchange3d_cuda_aware_mpi(d_aam, im, jm, kbm1);

		/*
		checkCudaErrors(cudaMemcpy(advx, d_advx, kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(advy, d_advy, kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(drhox, d_drhox, kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(drhoy, d_drhoy, kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(rho, d_rho, kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		*/

		/*
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					f_advx[k][j][i] = advx[k][j][i];
					f_advy[k][j][i] = advy[k][j][i];
					f_drhox[k][j][i] = drhox[k][j][i];
					f_drhoy[k][j][i] = drhoy[k][j][i];

					f_rho[k][j][i] = rho[k][j][i];
					f_aam[k][j][i] = aam[k][j][i];
				}
			}
		}
		*/
	}

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_momentum3d);
		momentum3d_time += time_consumed(&start_momentum3d, 
									     &end_momentum3d);
#endif

	return;
}




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
				int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	int kbm1 = kb-1;
	//int jmm1 = jm-1;
	//int imm1 = im-1;

	/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			adx2d[j][i] = 0.0f;
			ady2d[j][i] = 0.0f;
			drx2d[j][i] = 0.0f;
			dry2d[j][i] = 0.0f;
			aam2d[j][i] = 0.0f;
		}
	}
	*/

	if (j < jm && i < im){
		adx2d[j_off+i] = 0;
		ady2d[j_off+i] = 0;
		drx2d[j_off+i] = 0;
		dry2d[j_off+i] = 0;
		aam2d[j_off+i] = 0;
	}
	
	/*
	for(k = 0; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				adx2d[j][i] = adx2d[j][i]+advx[k][j][i]*dz[k];
				ady2d[j][i] = ady2d[j][i]+advy[k][j][i]*dz[k];
				drx2d[j][i] = drx2d[j][i]+drhox[k][j][i]*dz[k];
				dry2d[j][i] = dry2d[j][i]+drhoy[k][j][i]*dz[k];
				aam2d[j][i] = aam2d[j][i]+aam[k][j][i]*dz[k];
			}
		}
	}
	*/

	if (j < jm && i < im){
		for (k = 0; k < kbm1; k++){
			adx2d[j_off+i] += advx[k_off+j_off+i]*dz[k];
			ady2d[j_off+i] += advy[k_off+j_off+i]*dz[k];
			drx2d[j_off+i] += drhox[k_off+j_off+i]*dz[k];
			dry2d[j_off+i] += drhoy[k_off+j_off+i]*dz[k];
			aam2d[j_off+i] += aam[k_off+j_off+i]*dz[k];
		}
	}
}

__global__ void
mode_interaction_gpu_kernel_1(float * __restrict__ adx2d, 
							  float * __restrict__ ady2d,
							  const float * __restrict__ advua, 
							  const float * __restrict__ advva,
							  int jm, int im){

	//int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	//int kbm1 = kb-1;
	//int jmm1 = jm-1;
	//int imm1 = im-1;

	/*
	for(j = 0;  j < jm; j++){
		for(i = 0; i < im; i++){
			adx2d[j][i] = adx2d[j][i]-advua[j][i];
			ady2d[j][i] = ady2d[j][i]-advva[j][i]; 
		}
	}
	*/

	if (j < jm && i < im){
		adx2d[j_off+i] -= advua[j_off+i];	
		ady2d[j_off+i] -= advva[j_off+i];	
	}

}

__global__ void
mode_interaction_gpu_kernel_2(float * __restrict__ egf, 
							  const float * __restrict__ el,
							  float * __restrict__ utf, 
							  float * __restrict__ vtf, 
							  const float * __restrict__ ua, 
							  const float * __restrict__ va,
							  const float * __restrict__ d,
							  float ispi, float isp2i,
							  int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			egf[j][i] = el[j][i]*ispi;	
		}
	}
	*/

	if (j < jm && i < im){
		egf[j_off+i] = el[j_off+i]*ispi;	
	}

	/*
	for(j = 0; j < jm; j++){
		for(i = 1; i < im; i++){
			utf[j][i] = ua[j][i]*(d[j][i]+d[j][i-1])*isp2i;
		}
	}
	*/
	if (j < jm && i > 0 && i < im){
		utf[j_off+i] = ua[j_off+i]*(d[j_off+i]+d[j_off+(i-1)])*isp2i;
	}

	/*
	for(j = 1; j < jm; j++){
		for(i = 0; i < im; i++){
			vtf[j][i] = va[j][i]*(d[j][i]+d[j-1][i])*isp2i;
		}	
	}
	*/

	if (j > 0 && j < jm && i < im){
		vtf[j_off+i] = va[j_off+i]*(d[j_off+i]+d[j_1_off+i])*isp2i;
	}
}

void mode_interaction_gpu(){

	//int i,j,k;
#ifndef TIME_DISABLE
	struct timeval start_mode_interaction,
				   end_mode_interaction;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_mode_interaction);
#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	/*
	//kernel_0
	checkCudaErrors(cudaMemcpy(d_adx2d, adx2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ady2d, ady2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drx2d, drx2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dry2d, dry2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam2d, aam2d, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_advx, advx, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advy, advy, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drhox, drhox, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drhoy, drhoy, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	
	//advave
	//checkCudaErrors(cudaMemcpy(d_fluxua, fluxua, jm*im*sizeof(float), 
	//					cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_fluxva, fluxva, jm*im*sizeof(float), 
	//					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uab, uab, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vab, vab, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ua, ua, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_va, va, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//kernel_1
	checkCudaErrors(cudaMemcpy(d_advua, advua, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advva, advva, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//kernel_2
	checkCudaErrors(cudaMemcpy(d_el, el, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d, d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	*/
	
	//checkCudaErrors(cudaDeviceSynchronize());
	
	//advave
	if(mode != 2){

		/*
		//kernel_0
		checkCudaErrors(cudaMemcpy(d_advx, advx, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_advy, advy, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_drhox, drhox, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_drhoy, drhoy, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));

		//advave
		checkCudaErrors(cudaMemcpy(d_d, d, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_uab, uab, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_aam2d, aam2d, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vab, vab, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));

		//kernel_1
		checkCudaErrors(cudaMemcpy(d_advua, advua, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_advva, advva, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
		*/

		//modidy -adx2d, -ady2d, -drx2d, -dry2d, -aam2d
		mode_interaction_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
				d_adx2d, d_advx, d_ady2d, d_advy, d_drx2d, d_drhox,
				d_dry2d, d_drhoy, d_aam2d, d_aam, 
				d_dz, kb, jm, im);
		
		//checkCudaErrors(cudaDeviceSynchronize());
		

		/*
        advave_(advua,d,ua,va,fluxua,fluxva,uab,aam2d,
				vab,advva,wubot,wvbot);
		*/

		//modify -advua, -advva, -fluxua, -fluxva, 
		//		 +wubot, +wubot(may not modify them)
	
		
		/*
		advave_gpu(d_advua, d_advva, d_fluxua, d_fluxva, 
				   d_wubot, d_wvbot,
				   d_d, d_aam2d, d_ua, d_va, d_uab, d_vab);
		*/
		advave_gpu();
		
		/*
		checkCudaErrors(cudaMemcpy(advua, d_advua, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		
		checkCudaErrors(cudaMemcpy(advva, d_advva, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		*/
		//checkCudaErrors(cudaDeviceSynchronize());
		
		/*
		advave_gpu(advua, d, ua, va, fluxua, fluxva, 
				   uab, aam2d, vab, advva, wubot, wvbot);
		*/

		//modify -adx2d, -ady2d;
		
		mode_interaction_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
				d_adx2d, d_ady2d, d_advua, d_advva, 
				jm, im);
		

		//checkCudaErrors(cudaDeviceSynchronize());
		
		

		/*
		checkCudaErrors(cudaMemcpy(adx2d, d_adx2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(ady2d, d_ady2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		*/
		/*
		//kernel_0
		checkCudaErrors(cudaMemcpy(adx2d, d_adx2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(ady2d, d_ady2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(drx2d, d_drx2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(dry2d, d_dry2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(aam2d, d_aam2d, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));

		//advave
		checkCudaErrors(cudaMemcpy(advua, d_advua, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(advva, d_advva, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(fluxua, d_fluxua, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(fluxva, d_fluxva, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(wubot, d_wubot, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
		*/

	}
    //modify -egf, -utf, -vtf 
	
	mode_interaction_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
			d_egf, d_el, d_utf, d_vtf, d_ua, d_va, d_d, 
			ispi, isp2i, jm, im);
	
	
	/*
	//kernel_0&1
	checkCudaErrors(cudaMemcpy(adx2d, d_adx2d, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ady2d, d_ady2d, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(drx2d, d_drx2d, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(dry2d, d_dry2d, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(aam2d, d_aam2d, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

	//advave
	checkCudaErrors(cudaMemcpy(advua, d_advua, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(advva, d_advva, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(fluxua, d_fluxua, jm*im*sizeof(float), 
	//					cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(fluxva, d_fluxva, jm*im*sizeof(float), 
	//					cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(wubot, d_wubot, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	
	checkCudaErrors(cudaMemcpy(egf, d_egf, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(utf, d_utf, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vtf, d_vtf, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	*/
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_mode_interaction);
		mode_interaction_time += time_consumed(&start_mode_interaction, 
									           &end_mode_interaction);
#endif
	
	return;
}


__global__ void
mode_external_gpu_kernel_0(float * __restrict__ fluxua, 
						   float * __restrict__ fluxva,
						   const float * __restrict__ ua, 
						   const float * __restrict__ va,
						   const float * __restrict__ d, 
						   const float * __restrict__ dx, 
						   const float * __restrict__ dy,
						   int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	//int jmm1 = jm-1;
	//int imm1 = im-1;

	/*
	for(j = 1; j < jm; j++){
		for(i = 1; i < im; i++){
			fluxua[j][i] = 0.25f*(d[j][i]+d[j][i-1])
				*(dy[j][i]+dy[j][i-1])*ua[j][i];
			fluxva[j][i] = 0.25f*(d[j][i]+d[j-1][i])
				*(dx[j][i]+dx[j-1][i])*va[j][i];
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		fluxua[j_off+i] = 0.25f*(d[j_off+i]+d[j_off+(i-1)])
							   *(dy[j_off+i]+dy[j_off+(i-1)])
							   *ua[j_off+i];

		fluxva[j_off+i] = 0.25f*(d[j_off+i]+d[j_1_off+i])
							   *(dx[j_off+i]+dx[j_1_off+i])
							   *va[j_off+i];
	}

}

__global__ void
mode_external_gpu_kernel_1(const float * __restrict__ fluxua, 
						   const float * __restrict__ fluxva,
						   float * __restrict__ elf, 
						   const float * __restrict__ elb,
						   const float * __restrict__ vfluxf, 
						   const float * __restrict__ art, 
						   float dte2, int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	int jmm1 = jm-1;
	int imm1 = im-1;

	/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			elf[j][i] = elb[j][i]
						+dte2*(-(fluxua[j][i+1]-fluxua[j][i]
									+fluxva[j+1][i]-fluxva[j][i])/art[j][i]
								-vfluxf[j][i]);
		}
	}
	*/

	
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		elf[j_off+i] = elb[j_off+i]
					  +dte2*(-(fluxua[j_off+(i+1)]-fluxua[j_off+i]
							     +fluxva[j_A1_off+i]-fluxva[j_off+i])
							   /art[j_off+i] 
							 -vfluxf[j_off+i]);
	}
	
}

__global__ void
mode_external_inner_gpu_kernel_1(const float * __restrict__ fluxua, 
						   const float * __restrict__ fluxva,
						   float * __restrict__ elf, 
						   const float * __restrict__ elb,
						   const float * __restrict__ vfluxf, 
						   const float * __restrict__ art, 
						   const float * __restrict__ fsm,
						   float dte2, int jm, int im){

	const int j = blockDim.y*blockIdx.y+threadIdx.y+33;
	const int i = blockDim.x*blockIdx.x+threadIdx.x+33;

	if (j < jm-33 && i < im-33){
		elf[j_off+i] = elb[j_off+i]
					  +dte2*(-(fluxua[j_off+(i+1)]-fluxua[j_off+i]
							     +fluxva[j_A1_off+i]-fluxva[j_off+i])
							   /art[j_off+i] 
							 -vfluxf[j_off+i]);
		elf[j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
mode_external_ew_gpu_kernel_1(const float * __restrict__ fluxua, 
						   const float * __restrict__ fluxva,
						   float * __restrict__ elf, 
						   const float * __restrict__ elb,
						   const float * __restrict__ vfluxf, 
						   const float * __restrict__ art, 
						   const float * __restrict__ fsm,
						   float dte2, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	if (j < jm-1){
		elf[j_off+i] = elb[j_off+i]
					  +dte2*(-(fluxua[j_off+(i+1)]-fluxua[j_off+i]
							     +fluxva[j_A1_off+i]-fluxva[j_off+i])
							   /art[j_off+i] 
							 -vfluxf[j_off+i]);
		//elf[j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
mode_external_sn_gpu_kernel_1(const float * __restrict__ fluxua, 
						   const float * __restrict__ fluxva,
						   float * __restrict__ elf, 
						   const float * __restrict__ elb,
						   const float * __restrict__ vfluxf, 
						   const float * __restrict__ art, 
						   const float * __restrict__ fsm,
						   float dte2, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){ 
		elf[j_off+i] = elb[j_off+i]
					  +dte2*(-(fluxua[j_off+(i+1)]-fluxua[j_off+i]
							     +fluxva[j_A1_off+i]-fluxva[j_off+i])
							   /art[j_off+i] 
							 -vfluxf[j_off+i]);
		//elf[j_off+i] *= fsm[j_off+i];
	}
}

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
						   int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	int jmm1 = jm-1;
	int imm1 = im-1;
	float tmp;

	/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < im; i++){
			uaf[j][i] = adx2d[j][i]+advua[j][i]-aru[j][i]*0.25f
							*(cor[j][i]*d[j][i]*(va[j+1][i]+va[j][i])
								+cor[j][i-1]*d[j][i-1]*(va[j+1][i-1]+va[j][i-1]))
						+0.25f*grav*(dy[j][i]+dy[j][i-1])
							*(d[j][i]+d[j][i-1])
							*((1.0f-2.0f*alpha)*(el[j][i]-el[j][i-1])
								+alpha*(elb[j][i]-elb[j][i-1]
									+elf[j][i]-elf[j][i-1])
								+e_atmos[j][i]-e_atmos[j][i-1])
						+drx2d[j][i]+aru[j][i]*(wusurf[j][i]-wubot[j][i]);
		}
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < im){
		tmp = adx2d[j_off+i]+advua[j_off+i]
			 -aru[j_off+i]*0.25f
					*(cor[j_off+i]*d[j_off+i]
						*(va[j_A1_off+i]+va[j_off+i])
					 +cor[j_off+(i-1)]*d[j_off+(i-1)]
						*(va[j_A1_off+(i-1)]+va[j_off+(i-1)])) 
			 +0.25f*grav*(dy[j_off+i]+dy[j_off+(i-1)])
				   *(d[j_off+i]+d[j_off+(i-1)])
				   *((1.0f-2.0f*alpha)
						*(el[j_off+i]-el[j_off+(i-1)])
					+alpha*(elb[j_off+i]-elb[j_off+(i-1)]
						   +elf[j_off+i]-elf[j_off+(i-1)])
					+e_atmos[j_off+i]-e_atmos[j_off+(i-1)])
			 +drx2d[j_off+i]
			 +aru[j_off+i]*(wusurf[j_off+i]-wubot[j_off+i]);
	}
	*/

	/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < im; i++){
			uaf[j][i] = ((h[j][i]+elb[j][i]+h[j][i-1]+elb[j][i-1])
								*aru[j][i]*uab[j][i]
							-4.0f*dte*uaf[j][i])
						/((h[j][i]+elf[j][i]+h[j][i-1]+elf[j][i-1])
							*aru[j][i]);
		}
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < im){
		uaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_off+(i-1)]+elb[j_off+(i-1)])
						  *aru[j_off+i]*uab[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_off+(i-1)]+elf[j_off+(i-1)])
						 *aru[j_off+i]);
	}
	*/

	/*
	for(j = 1; j < jm; j++){
		for(i = 1; i < imm1; i++){
			vaf[j][i] = ady2d[j][i]+advva[j][i]
							+arv[j][i]*0.25f
								*(cor[j][i]*d[j][i]*(ua[j][i+1]+ua[j][i])
									+cor[j-1][i]*d[j-1][i]*(ua[j-1][i+1]+ua[j-1][i]))
							+0.25f*grav*(dx[j][i]+dx[j-1][i])
								*(d[j][i]+d[j-1][i])
								*((1.0f-2.0f*alpha)*(el[j][i]-el[j-1][i])
									+alpha*(elb[j][i]-elb[j-1][i]
										+elf[j][i]-elf[j-1][i])
									+e_atmos[j][i]-e_atmos[j-1][i])
							+dry2d[j][i]+arv[j][i]*(wvsurf[j][i]-wvbot[j][i]);
		}
	}
	*/

	/*
	if (j > 0 && j < jm && i > 0 && i < imm1){
		tmp = ady2d[j_off+i]+advva[j_off+i]
			 +arv[j_off+i]*0.25f
				*(cor[j_off+i]*d[j_off+i]
					*(ua[j_off+(i+1)]+ua[j_off+i])
				 +cor[j_1_off+i]*d[j_1_off+i]
					*(ua[j_1_off+(i+1)]+ua[j_1_off+i]))
			 +0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
				*(d[j_off+i]+d[j_1_off+i])
				*((1.0f-2.0f*alpha)
					*(el[j_off+i]-el[j_1_off+i])
				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
							+elf[j_off+i]-elf[j_1_off+i])
				  +e_atmos[j_off+i]-e_atmos[j_1_off+i])
			 +dry2d[j_off+i]
			 +arv[j_off+i]*(wvsurf[j_off+i]-wvbot[j_off+i]);
							
	}
	*/

	
	/*
	for(j = 1; j < jm; j++){
		for(i = 1; i < imm1; i++){
			vaf[j][i] = ((h[j][i]+elb[j][i]+h[j-1][i]+elb[j-1][i])
								*vab[j][i]*arv[j][i]
							-4.0f*dte*vaf[j][i])
						/((h[j][i]+elf[j][i]+h[j-1][i]+elf[j-1][i])
							*arv[j][i]);
		}
	}
	*/

	/*
	if (j > 0 && j < jm && i > 0 && i < imm1){
		vaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_1_off+i]+elb[j_1_off+i])
					     *vab[j_off+i]*arv[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_1_off+i]+elf[j_1_off+i])
						 *arv[j_off+i]);
	}
	*/

	if (j > 0 && j < jmm1 && i > 0 && i < im){
		tmp = adx2d[j_off+i]+advua[j_off+i]
			 -aru[j_off+i]*0.25f
					*(cor[j_off+i]*d[j_off+i]
						*(va[j_A1_off+i]+va[j_off+i])
					 +cor[j_off+(i-1)]*d[j_off+(i-1)]
						*(va[j_A1_off+(i-1)]+va[j_off+(i-1)])) 
			 +0.25f*grav*(dy[j_off+i]+dy[j_off+(i-1)])
				   *(d[j_off+i]+d[j_off+(i-1)])
				   *((1.0f-2.0f*alpha)
						*(el[j_off+i]-el[j_off+(i-1)])
					+alpha*(elb[j_off+i]-elb[j_off+(i-1)]
						   +elf[j_off+i]-elf[j_off+(i-1)])
					+e_atmos[j_off+i]-e_atmos[j_off+(i-1)])
			 +drx2d[j_off+i]
			 +aru[j_off+i]*(wusurf[j_off+i]-wubot[j_off+i]);
	}

	if (j > 0 && j < jmm1 && i > 0 && i < im){
		uaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_off+(i-1)]+elb[j_off+(i-1)])
						  *aru[j_off+i]*uab[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_off+(i-1)]+elf[j_off+(i-1)])
						 *aru[j_off+i]);
	}

	if (j > 0 && j < jm && i > 0 && i < imm1){
		tmp = ady2d[j_off+i]+advva[j_off+i]
			 +arv[j_off+i]*0.25f
				*(cor[j_off+i]*d[j_off+i]
					*(ua[j_off+(i+1)]+ua[j_off+i])
				 +cor[j_1_off+i]*d[j_1_off+i]
					*(ua[j_1_off+(i+1)]+ua[j_1_off+i]))
			 +0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
				*(d[j_off+i]+d[j_1_off+i])
				*((1.0f-2.0f*alpha)
					*(el[j_off+i]-el[j_1_off+i])
				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
							+elf[j_off+i]-elf[j_1_off+i])
				  +e_atmos[j_off+i]-e_atmos[j_1_off+i])
			 +dry2d[j_off+i]
			 +arv[j_off+i]*(wvsurf[j_off+i]-wvbot[j_off+i]);
							
	}

	if (j > 0 && j < jm && i > 0 && i < imm1){
		vaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_1_off+i]+elb[j_1_off+i])
					     *vab[j_off+i]*arv[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_1_off+i]+elf[j_1_off+i])
						 *arv[j_off+i]);
	}

	return;
}

__global__ void
mode_external_inner_gpu_kernel_2(float * __restrict__ uaf, 
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
						   const float * __restrict__ dum, 
						   const float * __restrict__ dvm, 
						   const float * __restrict__ dx, 
						   const float * __restrict__ dy,
						   const float * __restrict__ aru, 
						   const float * __restrict__ arv, 
						   const float * __restrict__ cor, 
						   const float * __restrict__ h,
						   float grav, float alpha, float dte,
						   int jm, int im){

	const int j = blockDim.y*blockIdx.y+threadIdx.y+33;
	const int i = blockDim.x*blockIdx.x+threadIdx.x+33;

	float tmp;

	if (j < jm-33 && i < im-33){
		tmp = adx2d[j_off+i]+advua[j_off+i]
			 -aru[j_off+i]*0.25f
					*(cor[j_off+i]*d[j_off+i]
						*(va[j_A1_off+i]+va[j_off+i])
					 +cor[j_off+(i-1)]*d[j_off+(i-1)]
						*(va[j_A1_off+(i-1)]+va[j_off+(i-1)])) 
			 +0.25f*grav*(dy[j_off+i]+dy[j_off+(i-1)])
				   *(d[j_off+i]+d[j_off+(i-1)])
				   *((1.0f-2.0f*alpha)
						*(el[j_off+i]-el[j_off+(i-1)])
					+alpha*(elb[j_off+i]-elb[j_off+(i-1)]
						   +elf[j_off+i]-elf[j_off+(i-1)])
					+e_atmos[j_off+i]-e_atmos[j_off+(i-1)])
			 +drx2d[j_off+i]
			 +aru[j_off+i]*(wusurf[j_off+i]-wubot[j_off+i]);

		uaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_off+(i-1)]+elb[j_off+(i-1)])
						  *aru[j_off+i]*uab[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_off+(i-1)]+elf[j_off+(i-1)])
						 *aru[j_off+i]);

		uaf[j_off+i] *= dum[j_off+i];

		tmp = ady2d[j_off+i]+advva[j_off+i]
			 +arv[j_off+i]*0.25f
				*(cor[j_off+i]*d[j_off+i]
					*(ua[j_off+(i+1)]+ua[j_off+i])
				 +cor[j_1_off+i]*d[j_1_off+i]
					*(ua[j_1_off+(i+1)]+ua[j_1_off+i]))
			 +0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
				*(d[j_off+i]+d[j_1_off+i])
				*((1.0f-2.0f*alpha)
					*(el[j_off+i]-el[j_1_off+i])
				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
							+elf[j_off+i]-elf[j_1_off+i])
				  +e_atmos[j_off+i]-e_atmos[j_1_off+i])
			 +dry2d[j_off+i]
			 +arv[j_off+i]*(wvsurf[j_off+i]-wvbot[j_off+i]);
							
		vaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_1_off+i]+elb[j_1_off+i])
					     *vab[j_off+i]*arv[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_1_off+i]+elf[j_1_off+i])
						 *arv[j_off+i]);
		vaf[j_off+i] *= dvm[j_off+i];

	}

	return;
}

__global__ void
mode_external_ew_gpu_kernel_2(float * __restrict__ uaf, 
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
						   const float * __restrict__ dum,
						   const float * __restrict__ dvm,
						   const float * __restrict__ dx, 
						   const float * __restrict__ dy,
						   const float * __restrict__ aru, 
						   const float * __restrict__ arv, 
						   const float * __restrict__ cor, 
						   const float * __restrict__ h,
						   float grav, float alpha, float dte,
						   int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	float tmp;

	if (j < jm-1){
		tmp = adx2d[j_off+i]+advua[j_off+i]
			 -aru[j_off+i]*0.25f
					*(cor[j_off+i]*d[j_off+i]
						*(va[j_A1_off+i]+va[j_off+i])
					 +cor[j_off+(i-1)]*d[j_off+(i-1)]
						*(va[j_A1_off+(i-1)]+va[j_off+(i-1)])) 
			 +0.25f*grav*(dy[j_off+i]+dy[j_off+(i-1)])
				   *(d[j_off+i]+d[j_off+(i-1)])
				   *((1.0f-2.0f*alpha)
						*(el[j_off+i]-el[j_off+(i-1)])
					+alpha*(elb[j_off+i]-elb[j_off+(i-1)]
						   +elf[j_off+i]-elf[j_off+(i-1)])
					+e_atmos[j_off+i]-e_atmos[j_off+(i-1)])
			 +drx2d[j_off+i]
			 +aru[j_off+i]*(wusurf[j_off+i]-wubot[j_off+i]);

		uaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_off+(i-1)]+elb[j_off+(i-1)])
						  *aru[j_off+i]*uab[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_off+(i-1)]+elf[j_off+(i-1)])
						 *aru[j_off+i]);


		tmp = ady2d[j_off+i]+advva[j_off+i]
			 +arv[j_off+i]*0.25f
				*(cor[j_off+i]*d[j_off+i]
					*(ua[j_off+(i+1)]+ua[j_off+i])
				 +cor[j_1_off+i]*d[j_1_off+i]
					*(ua[j_1_off+(i+1)]+ua[j_1_off+i]))
			 +0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
				*(d[j_off+i]+d[j_1_off+i])
				*((1.0f-2.0f*alpha)
					*(el[j_off+i]-el[j_1_off+i])
				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
							+elf[j_off+i]-elf[j_1_off+i])
				  +e_atmos[j_off+i]-e_atmos[j_1_off+i])
			 +dry2d[j_off+i]
			 +arv[j_off+i]*(wvsurf[j_off+i]-wvbot[j_off+i]);
							
		vaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_1_off+i]+elb[j_1_off+i])
					     *vab[j_off+i]*arv[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_1_off+i]+elf[j_1_off+i])
						 *arv[j_off+i]);

	}

	return;
}

__global__ void
mode_external_sn_gpu_kernel_2(float * __restrict__ uaf, 
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
						   const float * __restrict__ dum,
						   const float * __restrict__ dvm,
						   const float * __restrict__ dx, 
						   const float * __restrict__ dy,
						   const float * __restrict__ aru, 
						   const float * __restrict__ arv, 
						   const float * __restrict__ cor, 
						   const float * __restrict__ h,
						   float grav, float alpha, float dte,
						   int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	float tmp;

	if (i > 32 && i < im-33){ 
		tmp = adx2d[j_off+i]+advua[j_off+i]
			 -aru[j_off+i]*0.25f
					*(cor[j_off+i]*d[j_off+i]
						*(va[j_A1_off+i]+va[j_off+i])
					 +cor[j_off+(i-1)]*d[j_off+(i-1)]
						*(va[j_A1_off+(i-1)]+va[j_off+(i-1)])) 
			 +0.25f*grav*(dy[j_off+i]+dy[j_off+(i-1)])
				   *(d[j_off+i]+d[j_off+(i-1)])
				   *((1.0f-2.0f*alpha)
						*(el[j_off+i]-el[j_off+(i-1)])
					+alpha*(elb[j_off+i]-elb[j_off+(i-1)]
						   +elf[j_off+i]-elf[j_off+(i-1)])
					+e_atmos[j_off+i]-e_atmos[j_off+(i-1)])
			 +drx2d[j_off+i]
			 +aru[j_off+i]*(wusurf[j_off+i]-wubot[j_off+i]);

		uaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_off+(i-1)]+elb[j_off+(i-1)])
						  *aru[j_off+i]*uab[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_off+(i-1)]+elf[j_off+(i-1)])
						 *aru[j_off+i]);


		tmp = ady2d[j_off+i]+advva[j_off+i]
			 +arv[j_off+i]*0.25f
				*(cor[j_off+i]*d[j_off+i]
					*(ua[j_off+(i+1)]+ua[j_off+i])
				 +cor[j_1_off+i]*d[j_1_off+i]
					*(ua[j_1_off+(i+1)]+ua[j_1_off+i]))
			 +0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
				*(d[j_off+i]+d[j_1_off+i])
				*((1.0f-2.0f*alpha)
					*(el[j_off+i]-el[j_1_off+i])
				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
							+elf[j_off+i]-elf[j_1_off+i])
				  +e_atmos[j_off+i]-e_atmos[j_1_off+i])
			 +dry2d[j_off+i]
			 +arv[j_off+i]*(wvsurf[j_off+i]-wvbot[j_off+i]);
							
		vaf[j_off+i] = ((h[j_off+i]+elb[j_off+i]
							+h[j_1_off+i]+elb[j_1_off+i])
					     *vab[j_off+i]*arv[j_off+i]
						-4.0f*dte*tmp)
					   /((h[j_off+i]+elf[j_off+i]
							+h[j_1_off+i]+elf[j_1_off+i])
						 *arv[j_off+i]);
	}

	return;
}

__global__ void
mode_external_gpu_kernel_3(float * __restrict__ etf, 
						   //float * __restrict__ d,
						   //float * __restrict__ ua, 
						   //float * __restrict__ uab, 
						   //const float * __restrict__ uaf,
						   //float * __restrict__ va, 
						   //float * __restrict__ vab, 
						   //const float * __restrict__ vaf,
						   //float * __restrict__ el, 
						   //float * __restrict__ elb, 
						   const float * __restrict__ elf,
						   const float * __restrict__ fsm, 
						   //const float * __restrict__ h,
						   int iext, int isplit, float smoth,
						   int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	//int jmm1 = jm-1;
	//int imm1 = im-1;

	if(iext == (isplit-2)){
		/*
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				etf[j][i] = 0.25f*smoth*elf[j][i];
			}
		}
		*/
		
		if (j < jm && i < im){
			etf[j_off+i] = 0.25f*smoth*elf[j_off+i];
		}
		
	}
    else if(iext == (isplit-1)){
		/*
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				etf[j][i] = etf[j][i]+0.5f*(1.0f-0.5f*smoth)*elf[j][i];
			}
		}
		*/
		
		if (j < jm && i < im){
			etf[j_off+i] = etf[j_off+i]+0.5f*(1.0f-0.5f*smoth)*elf[j_off+i];
		}
    }
    else if(iext == isplit){
		/*
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				etf[j][i] = (etf[j][i]+0.5f*elf[j][i])*fsm[j][i];
			}
		}
		*/
	
		if (j < jm && i < im){
			etf[j_off+i] = (etf[j_off+i]+0.5f*elf[j_off+i])*fsm[j_off+i];
		}
		
    }
//! apply filter to remove time split

	/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			ua[j][i] = ua[j][i]+0.5f*smoth*(uab[j][i]-2.0f*ua[j][i]+uaf[j][i]);
			va[j][i] = va[j][i]+0.5f*smoth*(vab[j][i]-2.0f*va[j][i]+vaf[j][i]);
			el[j][i] = el[j][i]+0.5f*smoth*(elb[j][i]-2.0f*el[j][i]+elf[j][i]);
			elb[j][i] = el[j][i];
			el[j][i] = elf[j][i];
			d[j][i] = h[j][i]+el[j][i];
			uab[j][i] = ua[j][i];
			ua[j][i] = uaf[j][i];
			vab[j][i] = va[j][i];
			va[j][i] = vaf[j][i];
		}
	}
	*/

	
	//if (j < jm && i < im){
	//	ua[j_off+i] = ua[j_off+i]
	//				 +0.5f*smoth
	//					*(uab[j_off+i]-2.0f*ua[j_off+i]+uaf[j_off+i]);	
	//	va[j_off+i] = va[j_off+i]
	//				 +0.5f*smoth
	//					*(vab[j_off+i]-2.0f*va[j_off+i]+vaf[j_off+i]);	
	//	el[j_off+i] = el[j_off+i]
	//				 +0.5f*smoth
	//					*(elb[j_off+i]-2.0f*el[j_off+i]+elf[j_off+i]);	
	//	elb[j_off+i] = el[j_off+i];
	//	el[j_off+i] = elf[j_off+i];
	//	d[j_off+i] = h[j_off+i]+el[j_off+i];
	//	uab[j_off+i] = ua[j_off+i];
	//	ua[j_off+i] = uaf[j_off+i];
	//	vab[j_off+i] = va[j_off+i];
	//	va[j_off+i] = vaf[j_off+i];
	//}
	
}

__global__ void
mode_external_gpu_kernel_4(//float * __restrict__ etf, 
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
						   //const float * __restrict__ fsm, 
						   const float * __restrict__ h,
						   float smoth,
						   int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	if (j < jm && i < im){
		ua[j_off+i] = ua[j_off+i]
					 +0.5f*smoth
						*(uab[j_off+i]-2.0f*ua[j_off+i]+uaf[j_off+i]);	
		va[j_off+i] = va[j_off+i]
					 +0.5f*smoth
						*(vab[j_off+i]-2.0f*va[j_off+i]+vaf[j_off+i]);	
		el[j_off+i] = el[j_off+i]
					 +0.5f*smoth
						*(elb[j_off+i]-2.0f*el[j_off+i]+elf[j_off+i]);	
		elb[j_off+i] = el[j_off+i];
		el[j_off+i] = elf[j_off+i];
		d[j_off+i] = h[j_off+i]+el[j_off+i];
		uab[j_off+i] = ua[j_off+i];
		ua[j_off+i] = uaf[j_off+i];
		vab[j_off+i] = va[j_off+i];
		va[j_off+i] = vaf[j_off+i];
	}
}

__global__ void
mode_external_gpu_kernel_5(const float * __restrict__ d,
						   float * __restrict__ utf, 
						   float * __restrict__ vtf, 
						   float * __restrict__ egf, 
						   const float * __restrict__ ua, 
						   const float * __restrict__ va, 
						   const float * __restrict__ el, 
						   float ispi, float isp2i,
						   int iext, int isplit,
						   int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	if(iext != isplit){
		/*
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				egf[j][i] = egf[j][i]+el[j][i]*ispi;
			}
		}
		for(j = 0; j < jm; j++){
			for(i = 1; i < im; i++){
				utf[j][i] = utf[j][i]+ua[j][i]*(d[j][i]+d[j][i-1])*isp2i;
			}
		}
		for(j = 1; j < jm; j++){
			for(i = 0; i < im; i++){
				vtf[j][i] = vtf[j][i]+va[j][i]*(d[j][i]+d[j-1][i])*isp2i;
			}
		}
		*/

		
		if (j < jm && i < im){
			egf[j_off+i] = egf[j_off+i]+el[j_off+i]*ispi;
		}
		if (j < jm && i > 0 && i < im){
			utf[j_off+i] = utf[j_off+i]
						  +ua[j_off+i]*(d[j_off+i]+d[j_off+(i-1)])*isp2i;
		}
		if (j > 0 && j < jm && i < im){
			vtf[j_off+i] = vtf[j_off+i]
						  +va[j_off+i]*(d[j_off+i]+d[j_1_off+i])*isp2i;
		}
		
	}
}

void mode_external_gpu(){
	//int i,j;

#ifndef TIME_DISABLE
	struct timeval start_mode_external,
				   end_mode_external;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_mode_external);

#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	float *d_fluxua = d_2d_tmp0;
	float *d_fluxva = d_2d_tmp1;


	//modify -fluxua, -fluxva, -elf
	mode_external_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_fluxua, d_fluxva, d_ua, d_va, d_d, d_dx, d_dy, jm, im);

	mode_external_inner_gpu_kernel_1<<<blockPerGrid_inner, 
									   threadPerBlock_inner,
									   0, stream[0]>>>(
			d_fluxua, d_fluxva, d_elf, d_elb, d_vfluxf, 
			d_art, d_fsm, dte2, jm, im);

	mode_external_ew_gpu_kernel_1<<<blockPerGrid_ew_32, 
								    threadPerBlock_ew_32,
									0, stream[1]>>>(
			d_fluxua, d_fluxva, d_elf, d_elb, d_vfluxf, 
			d_art, d_fsm, dte2, jm, im);

	mode_external_sn_gpu_kernel_1<<<blockPerGrid_sn_32, 
								    threadPerBlock_sn_32,
									0, stream[2]>>>(
			d_fluxua, d_fluxva, d_elf, d_elb, d_vfluxf, 
			d_art, d_fsm, dte2, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));


	bcond_overlap(1, stream[1]);//modify elf boundary //just refernce fsm
	//bcond_gpu(1);//modify elf boundary //just refernce fsm

	//exchange2d_mpi_gpu(d_elf,im,jm);
	exchange2d_cudaUVA(d_elf, d_elf_east, d_elf_west, 
					   d_elf_south, d_elf_north,
					   stream[1], im,jm);

	//MPI_Barrier(pom_comm);
	//exchange2d_cuda_ipc(d_elf, d_elf_east, d_elf_west, 
	//					stream[1], im,jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);
	checkCudaErrors(cudaStreamSynchronize(stream[0]));
	
	if ((iext % ispadv) == 0){
		//advave_cbak_(advua,d,ua,va,fluxua,fluxva,uab,aam2d,vab,advva,wubot,wvbot);
		//modify -advua, -advva, -fluxua, -fluxva, 
		//		 +wubot, +wubot(may not modify them)
		advave_gpu();
	}


	//modify -uaf, -vaf

	mode_external_ew_gpu_kernel_2<<<blockPerGrid_ew_32, 
									threadPerBlock_ew_32,
									0, stream[1]>>>(
			d_uaf, d_ua, d_uab, d_vaf, d_va, d_vab, d_elf, d_el, d_elb,
			d_adx2d, d_ady2d, d_advua, d_advva, d_drx2d, d_dry2d,  
			d_wusurf, d_wvsurf, d_wubot, d_wvbot, d_e_atmos, d_d, 
			d_dum, d_dvm, d_dx, d_dy, d_aru, d_arv, d_cor, d_h,
			grav, alpha, dte, jm, im);

	mode_external_sn_gpu_kernel_2<<<blockPerGrid_sn_32, 
									threadPerBlock_sn_32,
									0, stream[2]>>>(
			d_uaf, d_ua, d_uab, d_vaf, d_va, d_vab, d_elf, d_el, d_elb,
			d_adx2d, d_ady2d, d_advua, d_advva, d_drx2d, d_dry2d,  
			d_wusurf, d_wvsurf, d_wubot, d_wvbot, d_e_atmos, d_d, 
			d_dum, d_dvm, d_dx, d_dy, d_aru, d_arv, d_cor, d_h,
			grav, alpha, dte, jm, im);


	mode_external_inner_gpu_kernel_2<<<blockPerGrid_inner, 
									threadPerBlock_inner,
									0, stream[0]>>>(
			d_uaf, d_ua, d_uab, d_vaf, d_va, d_vab, d_elf, d_el, d_elb,
			d_adx2d, d_ady2d, d_advua, d_advva, d_drx2d, d_dry2d,  
			d_wusurf, d_wvsurf, d_wubot, d_wvbot, d_e_atmos, d_d, 
			d_dum, d_dvm, d_dx, d_dy, d_aru, d_arv, d_cor, d_h,
			grav, alpha, dte, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

    bcond_overlap(2, stream[1]);//modify uaf&vaf boundary condition

    //exchange2d_mpi_gpu(d_uaf,im,jm);
    //exchange2d_mpi_gpu(d_vaf,im,jm);

	exchange2d_cudaUVA(d_uaf, d_uaf_east, d_uaf_west, 
					   d_uaf_south, d_uaf_north,
					   stream[1], im,jm);

	exchange2d_cudaUVA(d_vaf, d_vaf_east, d_vaf_west, 
					   d_vaf_south, d_vaf_north,
					   stream[1], im,jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	//MPI_Barrier(pom_comm);
	//exchange2d_cuda_ipc(d_uaf, d_uaf_east, d_uaf_west, 
	//					stream[1], im,jm);
	//exchange2d_cuda_ipc(d_vaf, d_vaf_east, d_vaf_west, 
	//					stream[1], im,jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[0]));
	
	//if (calc_vort) vort_gpu();

	//modify +etf(may just -), +ua, +va, +el, +egf, +utf, +vtf, 
	//		 +elb, +uab, +vab
	//       -d 
	//mode_external_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
	//		d_etf, d_d, d_ua, d_uab, d_uaf,
	//		d_va, d_vab, d_vaf, d_el, d_elb, d_elf, d_fsm, d_h,
	//		iext, isplit, smoth, jm, im);

	mode_external_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
			d_etf, d_elf, d_fsm, 
			iext, isplit, smoth, jm, im);

	if (iext == isplit){
		if (calc_vort) vort_gpu();
	}

	mode_external_gpu_kernel_4<<<blockPerGrid, threadPerBlock>>>(
			d_d, d_ua, d_uab, d_uaf,
			d_va, d_vab, d_vaf, d_el, d_elb, d_elf, d_h,
			smoth, jm, im);

	//checkCudaErrors(cudaDeviceSynchronize());

	
	mode_external_gpu_kernel_5<<<blockPerGrid, threadPerBlock>>>(
			d_d, d_utf, d_vtf, d_egf, d_ua, d_va, d_el,
			ispi, isp2i, iext, isplit, jm, im);
	

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_mode_external);
		mode_external_time += time_consumed(&start_mode_external, 
									        &end_mode_external);
#endif
    return;
}

__global__ void
mode_internal_gpu_kernel_0(float * __restrict__ u, 
						   float * __restrict__ v, 
						   const float * __restrict__ utb, 
						   const float * __restrict__ utf, 
						   const float * __restrict__ vtb, 
						   const float * __restrict__ vtf, 
						   const float * __restrict__ dt,
						   const float * __restrict__ dz, 
						   int kb, int jm, int im){
	int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	int kbm1 = kb-1;
	//int jmm1 = jm-1;
	//int imm1 = im-1;
	float tps = 0;

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			tps[j][i] = 0.0f;	
		}
	}
	*/

	//0 needed badly for +=
	/*
	if (j < jm  && i < im){
		tps[j_off+i] = 0;	
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] += u[k][j][i]*dz[k];	
			}
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps[j_off+i] += u[k_off+j_off+i]*dz[k];	
		}
	}
	*/
	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps += u[k_off+j_off+i]*dz[k];	
		}
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 1; i < im; i++){
				u[k][j][i] = (u[k][j][i]-tps[j][i])+(utb[j][i]+utf[j][i])/(dt[j][i]+dt[j][i-1]);
			}
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j < jm && i > 0 && i < im){
			u[k_off+j_off+i] = (u[k_off+j_off+i]-tps[j_off+i]) +
									(utb[j_off+i]+utf[j_off+i])/(dt[j_off+i]+dt[j_off+(i-1)]);
		}
	}
	*/
	for (k = 0; k < kbm1; k++){
		if (j < jm && i > 0 && i < im){
			u[k_off+j_off+i] = (u[k_off+j_off+i]-tps)
							  +(utb[j_off+i]+utf[j_off+i])
									/(dt[j_off+i]+dt[j_off+(i-1)]);
		}
	}

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			tps[j][i] = 0.0f;	
		}
	}
	*/

	//0 needed badly for +=
	/*
	if (j < jm && i < im){
		tps[j_off+i] = 0;	
	}
	*/
	tps = 0;

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] += v[k][j][i]*dz[k];	
			}
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps[j_off+i] += v[k_off+j_off+i]*dz[k];	
		}
	}
	*/
	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps += v[k_off+j_off+i]*dz[k];	
		}
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 0; i < im; i++){
				v[k][j][i] = (v[k][j][i]-tps[j][i])+(vtb[j][i]+vtf[j][i])/(dt[j][i]+dt[j-1][i]);
			}
		}
	}
	*/
	/*
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jm && i < im){
			v[k_off+j_off+i] = (v[k_off+j_off+i]-tps[j_off+i]) +
									(vtb[j_off+i]+vtf[j_off+i])/(dt[j_off+i]+dt[j_1_off+i]);
		}
	}
	*/
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jm && i < im){
			v[k_off+j_off+i] = (v[k_off+j_off+i]-tps)
							  +(vtb[j_off+i]+vtf[j_off+i])
									/(dt[j_off+i]+dt[j_1_off+i]);
		}
	}
}

__global__ void 
mode_internal_gpu_kernel_1(float * __restrict__ q2, 
						   float * __restrict__ q2b,
						   float * __restrict__ q2l, 
						   float * __restrict__ q2lb,
						   const float * __restrict__ uf, 
						   const float * __restrict__ vf, 
						   float smoth,
						   int kb, int jm, int im){
	int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				q2[k][j][i] = q2[k][j][i]+0.5f*smoth*
											(uf[k][j][i]+q2b[k][j][i]-2.0f*q2[k][j][i]);	
				q2l[k][j][i] = q2l[k][j][i]+0.5f*smoth*
											(vf[k][j][i]+q2lb[k][j][i]-2.0f*q2l[k][j][i]);
				q2b[k][j][i] = q2[k][j][i];
				q2[k][j][i] = uf[k][j][i];
				q2lb[k][j][i] = q2l[k][j][i];
				q2l[k][j][i] = vf[k][j][i];
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			q2[k_off+j_off+i] = q2[k_off+j_off+i]
							   +0.5f*smoth
									*(uf[k_off+j_off+i]
									 +q2b[k_off+j_off+i]
									 -2.0f*q2[k_off+j_off+i]);

			q2l[k_off+j_off+i] = q2l[k_off+j_off+i]
								+0.5f*smoth
									 *(vf[k_off+j_off+i]
									  +q2lb[k_off+j_off+i]
									  -2.0f*q2l[k_off+j_off+i]);

			q2b[k_off+j_off+i] = q2[k_off+j_off+i];
			q2[k_off+j_off+i] = uf[k_off+j_off+i];
			q2lb[k_off+j_off+i] = q2l[k_off+j_off+i];
			q2l[k_off+j_off+i] = vf[k_off+j_off+i];
		}
	}
}

__global__ void
mode_internal_gpu_kernel_2(float * __restrict__ tb, 
						   float * __restrict__ t, 
						   float * __restrict__ uf,
						   float * __restrict__ sb, 
						   float * __restrict__ s, 
						   float * __restrict__ vf,
						   float smoth, 
						   int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				t[k][j][i] = t[k][j][i]+0.5f*smoth*
										(uf[k][j][i]+tb[k][j][i]-2.0f*t[k][j][i]);
				s[k][j][i] = s[k][j][i]+0.5f*smoth*
										(vf[k][j][i]+sb[k][j][i]-2.0f*s[k][j][i]);
				tb[k][j][i] = t[k][j][i];
				t[k][j][i] = uf[k][j][i];
				sb[k][j][i] = s[k][j][i];
				s[k][j][i] = vf[k][j][i];
			}
		}
	}
	*/
	
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			t[k_off+j_off+i] += 0.5f*smoth*(uf[k_off+j_off+i]+tb[k_off+j_off+i]-
												2.0f*t[k_off+j_off+i]);
			s[k_off+j_off+i] += 0.5f*smoth*(vf[k_off+j_off+i]+sb[k_off+j_off+i]-
												2.0f*s[k_off+j_off+i]);
			tb[k_off+j_off+i] = t[k_off+j_off+i];
			t[k_off+j_off+i] = uf[k_off+j_off+i];
			sb[k_off+j_off+i] = s[k_off+j_off+i];
			s[k_off+j_off+i] = vf[k_off+j_off+i];
		}
	}
}

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
						   int kb, int jm, int im){

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z+threadIdx.z;
#endif
	int k;

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	int kbm1 = kb-1;

#ifdef D3_BLOCK
	float tps = 0;

	if (k < kbm1 && j < jm && i < im){
		tps += (uf[k_off+j_off+i]
			   +ub[k_off+j_off+i]
			   -2.0f*u[k_off+j_off+i])*dz[k];
	}

	if (k < kbm1 && j < jm && i < im){
		u[k_off+j_off+i] += 0.5f*smoth
								*(uf[k_off+j_off+i]
								 +ub[k_off+j_off+i]
								 -2.0f*u[k_off+j_off+i]-tps);
	}
	
	tps = 0;

	if (k < kbm1 && j < jm && i < im){
		tps += (vf[k_off+j_off+i]
			   +vb[k_off+j_off+i]
			   -2.0f*v[k_off+j_off+i])*dz[k];
	}

	if (k < kbm1 && j < jm && i < im){
		v[k_off+j_off+i] += 0.5f*smoth
								*(vf[k_off+j_off+i]
								 +vb[k_off+j_off+i]
								 -2.0f*v[k_off+j_off+i]-tps);
	}
	
	
	if (k < kb && j < jm && i < im){
		ub[k_off+j_off+i] = u[k_off+j_off+i];	
		u[k_off+j_off+i] = uf[k_off+j_off+i];

		vb[k_off+j_off+i] = v[k_off+j_off+i];
		v[k_off+j_off+i] = vf[k_off+j_off+i];
	}
#endif 

#ifdef optimize
	float tps = 0;

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps += (uf[k_off+j_off+i]+ub[k_off+j_off+i] -
								2.0f*u[k_off+j_off+i])*dz[k];
		}
	}

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			u[k_off+j_off+i] += 0.5f*smoth*(uf[k_off+j_off+i]+ub[k_off+j_off+i] -
												2.0f*u[k_off+j_off+i]-tps);
		}
	}

	tps = 0;

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps += (vf[k_off+j_off+i]+vb[k_off+j_off+i] -
								2.0f*v[k_off+j_off+i])*dz[k];
		}
	}

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			v[k_off+j_off+i] += 0.5f*smoth*(vf[k_off+j_off+i]+vb[k_off+j_off+i] -
												2.0f*v[k_off+j_off+i]-tps);
		}
	}

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			ub[k_off+j_off+i] = u[k_off+j_off+i];	
			u[k_off+j_off+i] = uf[k_off+j_off+i];

			vb[k_off+j_off+i] = v[k_off+j_off+i];
			v[k_off+j_off+i] = vf[k_off+j_off+i];
		}
	}
#endif
	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			tps[j][i] = 0.0f;	
		}
	}
	*/
	if (j < jm && i < im){
		tps[j_off+i] = 0;	
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = tps[j][i]+(uf[k][j][i]+ub[k][j][i]-2.0f*u[k][j][i])*dz[k];
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps[j_off+i] += (uf[k_off+j_off+i]
							+ub[k_off+j_off+i]
							-2.0f*u[k_off+j_off+i])*dz[k];
		}
	}


	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				u[k][j][i] = u[k][j][i]+0.5f*smoth*
										 (uf[k][j][i]+ub[k][j][i]-2.0f*u[k][j][i]-tps[j][i]);
			}
		}
	}
	*/
	
	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			u[k_off+j_off+i] += 0.5f*smoth
									*(uf[k_off+j_off+i]
									 +ub[k_off+j_off+i]
									 -2.0f*u[k_off+j_off+i]
									 -tps[j_off+i]);
		}
	}
	

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			tps[j][i] = 0.0f;	
		}
	}
	*/

	if (j < jm && i < im){
		tps[j_off+i] = 0;	
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = tps[j][i]+(vf[k][j][i]+vb[k][j][i]-2.0f*v[k][j][i])*dz[k];
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			tps[j_off+i] += (vf[k_off+j_off+i]
							+vb[k_off+j_off+i]
							-2.0f*v[k_off+j_off+i])*dz[k];
		}
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				v[k][j][i] = v[k][j][i]+0.5f*smoth*
										 (vf[k][j][i]+vb[k][j][i]-2.0f*v[k][j][i]-tps[j][i]);
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j < jm && i < im){
			v[k_off+j_off+i] += 0.5f*smoth
									*(vf[k_off+j_off+i]
									 +vb[k_off+j_off+i]
									 -2.0f*v[k_off+j_off+i]
									 -tps[j_off+i]);
		}
	}
	

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				ub[k][j][i] = u[k][j][i];
				u[k][j][i] = uf[k][j][i];

				vb[k][j][i] = v[k][j][i];
				v[k][j][i] = vf[k][j][i];
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			ub[k_off+j_off+i] = u[k_off+j_off+i];	
			u[k_off+j_off+i] = uf[k_off+j_off+i];

			vb[k_off+j_off+i] = v[k_off+j_off+i];
			v[k_off+j_off+i] = vf[k_off+j_off+i];
		}
	}
	
	

}

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
						   int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			egb[j][i] = egf[j][i];
			etb[j][i] = et[j][i];
			et[j][i] = etf[j][i];

			dt[j][i] = h[j][i]+et[j][i];

			utb[j][i] = utf[j][i];
			vtb[j][i] = vtf[j][i];

			vfluxb[j][i] = vfluxf[j][i];
		}
	}
	*/

	if (j < jm && i < im){
		egb[j_off+i] = egf[j_off+i];
		etb[j_off+i] = et[j_off+i];
		et[j_off+i] = etf[j_off+i];

		dt[j_off+i] = h[j_off+i] + et[j_off+i];

		utb[j_off+i] = utf[j_off+i];
		vtb[j_off+i] = vtf[j_off+i];

		vfluxb[j_off+i] = vfluxf[j_off+i];
	}
}


void mode_internal_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_mode_internal,
				   end_mode_internal;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_mode_internal);
#endif

	//int i, j, k;
	//float tps[j_size][i_size];

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

#ifdef D3_BLOCK
	dim3 threadPerBlock_D3(block_i_3D, block_j_3D, block_k_3D);
	dim3 blockPerGrid_D3((im+block_i_3D-1)/block_i_3D, (jm+block_j_3D-1)/block_j_3D, (kb+block_k_3D-1)/block_k_3D);
#endif


	float *d_tps = d_3d_tmp0;
	
	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				
				u[k][j][i] = f_u[k][j][i];	
				v[k][j][i] = f_v[k][j][i];
				q2b[k][j][i] = f_q2b[k][j][i];
				q2[k][j][i] = f_q2[k][j][i];
				aam[k][j][i] = f_aam[k][j][i];
				q2lb[k][j][i] = f_q2lb[k][j][i];
				q2l[k][j][i] = f_q2l[k][j][i];
				

			
				t[k][j][i] = f_t[k][j][i];
				s[k][j][i] = f_s[k][j][i];
				rho[k][j][i] = f_rho[k][j][i];
				tb[k][j][i] = f_tb[k][j][i];
				sb[k][j][i] = f_sb[k][j][i];

				
				advx[k][j][i] = f_advx[k][j][i];
				advy[k][j][i] = f_advy[k][j][i];

				ub[k][j][i] = f_ub[k][j][i];
				vb[k][j][i] = f_vb[k][j][i];
			
				drhox[k][j][i] = f_drhox[k][j][i];
				drhoy[k][j][i] = f_drhoy[k][j][i];


				//km[k][j][i] = f_km[k][j][i];
				//kh[k][j][i] = f_kh[k][j][i];

			}
		}
	}
	*/


	/*
	
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			utb[j][i] = f_utb[j][i];
			utf[j][i] = f_utf[j][i];
			dt[j][i] = f_dt[j][i];
			vtb[j][i] = f_vtb[j][i];
			vtf[j][i] = f_vtf[j][i];
			etf[j][i] = f_etf[j][i];
			etb[j][i] = f_etb[j][i];
			vfluxb[j][i] = f_vfluxb[j][i];
			vfluxf[j][i] = f_vfluxf[j][i];
			wusurf[j][i] = f_wusurf[j][i];
			wvsurf[j][i] = f_wvsurf[j][i];
			wubot[j][i] = f_wubot[j][i];
			wvbot[j][i] = f_wvbot[j][i];
			egf[j][i] = f_egf[j][i];
			egb[j][i] = f_egb[j][i];
			e_atmos[j][i] = f_e_atmos[j][i];
			wtsurf[j][i] = f_wtsurf[j][i];
			wssurf[j][i] = f_wssurf[j][i];
			swrad[j][i] = f_swrad[j][i];
			et[j][i] = f_et[j][i];
		
		}
	}
	*/
	
	/*
	//kernel_0
	checkCudaErrors(cudaMemcpy(d_utb, utb, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_utf, utf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vtb, vtb, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vtf, vtf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	
	//vertvl
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfluxb, vfluxb, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfluxf, vfluxf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));

	//advave
	checkCudaErrors(cudaMemcpy(d_q2b, q2b, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2, q2, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2lb, q2lb, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2l, q2l, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dum, dum, jm*im*sizeof(float), 
						       cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dvm, dvm, jm*im*sizeof(float), 
						       cudaMemcpyHostToDevice));

	//profq
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_t, t, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_s, s, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rho, rho, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kh, kh, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kq, kq, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wusurf, wusurf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvsurf, wvsurf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));

	//advt1
	checkCudaErrors(cudaMemcpy(d_tb, tb, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sb, sb, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tsurf, tsurf, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));

	//proft
	checkCudaErrors(cudaMemcpy(d_wtsurf, wtsurf, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wssurf, wssurf, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ssurf, ssurf, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));

	//advu
	checkCudaErrors(cudaMemcpy(d_egb, egb, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));

	//profu
	checkCudaErrors(cudaMemcpy(d_ub, ub, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vb, vb, kb*jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cbc, cbc, jm*im*sizeof(float), 
						cudaMemcpyHostToDevice));

	//kernel_4
	checkCudaErrors(cudaMemcpy(d_egf, egf, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_et, et, jm*im*sizeof(float), 
							cudaMemcpyHostToDevice));
	*/

	//kernel_0
	////////////////////////////////////////////////////////////
	//2 advq, one for vf, one for uf
	////////////////////////////////////////////////////////////
	//profq
	
	
	

	
	///////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////
	//proft
	///////////////////////////////////////////////////////////
	//advu-uf
	///////////////////////////////////////////////////////////
	//advv-vf
	///////////////////////////////////////////////////////////
	//profu-uf
	///////////////////////////////////////////////////////////
	//profv-vf
	///////////////////////////////////////////////////////////
	//kernel_4
	//iint = *f_iint;

	if ((iint != 1 || time0 != 0.0f) && mode != 2){

		//modify ?u, ?v 
		//comment the boundary is not assigned in the below function
		//but after a test , the boundary is not important,
		//so we can choose not to copy-in u and v
		
		//mode_internal_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
		//		d_u, d_v, d_utb, d_utf, d_vtb, d_vtf, 
		//		d_tps_mode_internal, d_dt, d_dz, 
		//		kb, jm, im);
		
		mode_internal_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
				d_u, d_v, d_utb, d_utf, d_vtb, d_vtf, d_dt, d_dz, 
				kb, jm, im);

        //vertvl_(dt,u,v,vfluxb,vfluxf,w,etf,etb);//modified -w
		
		//vertvl_gpu(d_u, d_v, d_w, d_vfluxb, d_vfluxf, 
		//           d_etf, d_etb, d_dt);
		
		//vertvl_gpu();

		//bcond_gpu(5);//only modify w and reference fsm

		////exchange3d_mpi_gpu(d_w,im,jm,kb);
		//exchange3d_cuda_aware_mpi(d_w,im,jm,kb);


		vertvl_overlap_bcond(stream[0], stream[1], stream[2]);
		checkCudaErrors(cudaStreamSynchronize(stream[1]));
		checkCudaErrors(cudaStreamSynchronize(stream[2]));

		bcond_overlap(5, stream[1]);//only modify w and reference fsm

		//exchange3d_mpi_gpu(d_w,im,jm,kb);
		exchange3d_cudaUVA(d_w, d_w_east, d_w_west, 
						   d_w_south, d_w_north,
						   stream[1], im,jm,kb);

		//MPI_Barrier(pom_comm);
		//exchange3d_cuda_ipc(d_w, d_w_east, d_w_west, 
		//					stream[1], im,jm,kb);

		//checkCudaErrors(cudaStreamSynchronize(stream[1]));
		//MPI_Barrier(pom_comm);
		checkCudaErrors(cudaStreamSynchronize(stream[0]));

		//advq_(q2b,q2,uf,u,dt,v,aam,w,etb,etf);//only modify uf
		//only modify uf
		
		//advq_gpu(d_q2b, d_q2, d_uf, d_u, d_v, d_w, 
		//		 d_etb, d_etf, d_aam, d_dt);
		
		//advq_gpu(d_q2b, d_q2, d_uf); 


		//advq_(q2lb,q2l,vf,u,dt,v,aam,w,etb,etf);//only modify vf
		//only modify vf
		
		//advq_gpu(d_q2lb, d_q2l, d_vf, d_u, d_v, d_w, 
		//		 d_etb, d_etf, d_aam, d_dt);
		
		//advq_gpu(d_q2lb, d_q2l, d_vf);

		advq_fusion_gpu(d_q2b, d_q2, d_uf, 
						d_q2lb, d_q2l, d_vf);

		// modify:
		//			uf, vf, 
		//	+reference: kq, km, kh
		//	+reference:	q2b, q2lb
		//profq_(etf,wusurf,wvsurf,wubot,wvbot,q2b,q2lb,u,v,
		//			km,uf,vf,q2,dt,kh,t,s,rho);//

		
		//profq_gpu(d_uf, d_vf, d_km, d_kh, d_q2b, d_q2lb, 
		//		  d_wusurf, d_wvsurf, d_wubot, d_wvbot, 
		//		  d_u, d_v, d_t, d_s, 
		//		  d_dt, d_rho, d_q2, d_etf);
		

		//profq_gpu();

		//bcond_gpu(6); //only modify uf vf

		////exchange3d_mpi_gpu(d_uf+jm*im,im,jm,kbm1-1);
		////exchange3d_mpi_gpu(d_vf+jm*im,im,jm,kbm1-1);

		//exchange3d_cuda_aware_mpi(d_uf+jm*im,im,jm,kbm1-1);
		//exchange3d_cuda_aware_mpi(d_vf+jm*im,im,jm,kbm1-1);

		profq_overlap_bcond();
		bcond_overlap(6, stream[1]); //only modify uf vf

		//exchange3d_mpi_gpu(d_uf+jm*im,im,jm,kbm1-1);
		//exchange3d_mpi_gpu(d_vf+jm*im,im,jm,kbm1-1);

		exchange3d_cudaUVA(d_uf+jm*im, d_uf_east+jm*im, d_uf_west+jm*im,
						   d_uf_south+jm*im, d_uf_north+jm*im,
						   stream[1], im, jm, kbm1);

		exchange3d_cudaUVA(d_vf+jm*im, d_vf_east+jm*im, d_vf_west+jm*im,
						   d_vf_south+jm*im, d_vf_north+jm*im,
						   stream[1], im, jm, kbm1);

		//MPI_Barrier(pom_comm);
		//exchange3d_cuda_ipc(d_uf+jm*im, d_uf_east+jm*im, d_uf_west+jm*im,
		//					stream[1], im, jm, kbm1);
		//exchange3d_cuda_ipc(d_vf+jm*im, d_vf_east+jm*im, d_vf_west+jm*im,
		//					stream[1], im, jm, kbm1);
		//checkCudaErrors(cudaStreamSynchronize(stream[1]));
		//MPI_Barrier(pom_comm);

		checkCudaErrors(cudaStreamSynchronize(stream[0]));
		
		//modify +q2, +q2l, +q2b, +q2lb
		mode_internal_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
				d_q2, d_q2b, d_q2l, d_q2lb, d_uf, d_vf, 
				smoth, kb, jm, im);

		if (mode != 4){
			if (nadv == 1){
				//advt1_(tb,t,tclim,uf,dt,u,v,aam,w,etb,etf);//modify -uf +t tb
				
				//advt1_gpu(d_tb, d_t, d_tclim, d_uf, 
				//		  d_u, d_v, d_w, 
				//		  d_etb, d_etf, d_aam, d_dt);
				
				advt1_gpu(d_tb, d_t, d_tclim, d_uf, 'T'); 

				//advt1_(sb,s,sclim,vf,dt,u,v,aam,w,etb,etf);//modifuy -vf +s sb
				
				//advt1_gpu(d_sb, d_s, d_sclim, d_vf, 
				//	      d_u, d_v, d_w,
				//		  d_etb, d_etf, d_aam, d_dt);
				
				advt1_gpu(d_sb, d_s, d_sclim, d_vf, 'S'); 

			}else if(nadv == 2){ 

				//modify -uf +tb
				//and MPI uf 
				//advt2_(tb,t,tclim,uf,etb,u,v,etf,aam,w,dt);
				
				//advt2_gpu(d_tb, d_t, d_tclim, d_uf, 
				//		  d_u, d_v, d_w, 
				//		  d_etb, d_etf, d_aam, d_dt);
				
				//advt2_gpu(d_tb, d_t, d_tclim, d_uf, 'T');
				advt2_gpu(d_tb, d_t, d_tclim, d_uf, 
						  d_uf_east, d_uf_west, d_uf_south, d_uf_north,
						  'T');

				//modify -vf +sb
				//and MPI vf 
				//advt2_(sb,s,sclim,vf,etb,u,v,etf,aam,w,dt);//modify -vf +sb
				
				//advt2_gpu(d_sb, d_s, d_sclim, d_vf, 
				//		  d_u, d_v, d_w, 
				//		  d_etb, d_etf, d_aam, d_dt);
				
				//advt2_gpu(d_sb, d_s, d_sclim, d_vf, 'S'); 
				advt2_gpu(d_sb, d_s, d_sclim, d_vf, 
						  d_vf_east, d_vf_west, d_vf_south, d_vf_north,
						  'S'); 

			}else{
				error_status = 1;
				printf("Error: invalid value for nadv! Error find in File:%s, Func:%s, Line:%d\n",
						__FILE__, __func__, __LINE__);
			}

			//modify +uf
			//MPI uf
			//proft_(uf,wtsurf,tsurf,nbct,etf,kh,swrad);
			//proft_gpu(d_uf, d_wtsurf, d_tsurf, d_etf, d_kh, d_swrad, &nbct);
			
			//proft_gpu(d_uf, d_wtsurf, d_tsurf, d_etf, 
			//		  d_kh, d_swrad, nbct);
			
			//proft_gpu(d_uf, d_wtsurf, d_tsurf, nbct);
			//checkCudaErrors(cudaDeviceSynchronize());


			
			//modify +vf
			//MPI vf
			//proft_(vf,wssurf,ssurf,nbcs,etf,kh,swrad);
			//proft_gpu(d_vf, d_wssurf, d_ssurf, d_etf, d_kh, d_swrad, &nbcs);
			
			//proft_gpu(d_vf, d_wssurf, d_ssurf, d_etf, 
			//          d_kh, d_swrad, nbcs);
			
			//proft_gpu(d_vf, d_wssurf, d_ssurf, nbcs);

			//proft_fusion_gpu(d_uf, d_wtsurf, d_tsurf, nbct,
			//				 d_vf, d_wssurf, d_ssurf, nbcs);

			//bcond_gpu(4);//modify +uf +vf

			///////////////////////////////////////////////////////////
			//advt1 copy-backs contains advt2
			////exchange3d_mpi_gpu(d_uf,im,jm,kbm1);
			////exchange3d_mpi_gpu(d_vf,im,jm,kbm1);
			//exchange3d_cuda_aware_mpi(d_uf,im,jm,kbm1);
			//exchange3d_cuda_aware_mpi(d_vf,im,jm,kbm1);

			//proft_fusion_gpu(d_uf, d_wtsurf, d_tsurf, nbct,
			//				 d_vf, d_wssurf, d_ssurf, nbcs);

			proft_fusion_overlap_bcond(
							 d_uf, d_wtsurf, d_tsurf, nbct,
							 d_vf, d_wssurf, d_ssurf, nbcs,
							 stream[0], stream[1], stream[2]);

			checkCudaErrors(cudaStreamSynchronize(stream[1]));
			checkCudaErrors(cudaStreamSynchronize(stream[2]));

			bcond_overlap(4, stream[1]);//modify +uf +vf

			//exchange3d_mpi_gpu(d_uf,im,jm,kbm1);
			//exchange3d_mpi_gpu(d_vf,im,jm,kbm1);

			exchange3d_cudaUVA(d_uf, d_uf_east, d_uf_west, 
							   d_uf_south, d_uf_north,
							   stream[1], im,jm,kbm1);

			exchange3d_cudaUVA(d_vf, d_vf_east, d_vf_west,
							   d_vf_south, d_vf_north,
							   stream[1], im,jm,kbm1);

			//MPI_Barrier(pom_comm);
			//exchange3d_cuda_ipc(d_uf, d_uf_east, d_uf_west, 
			//					stream[1], im,jm,kbm1);
			//exchange3d_cuda_ipc(d_vf, d_vf_east, d_vf_west,
			//					stream[1], im,jm,kbm1);

			//checkCudaErrors(cudaStreamSynchronize(stream[1]));
			//MPI_Barrier(pom_comm);
			checkCudaErrors(cudaStreamSynchronize(stream[0]));

			//modify +t, +tb, +s, +sb
			mode_internal_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
					d_tb, d_t, d_uf, d_sb, d_s, d_vf, 
					smoth, kb, jm, im);

			//dens_(s,t,rho);//modify -rho
			dens_gpu(d_s, d_t, d_rho);
		}


		if (tracer_flag != 0){
			/*
			for (inb = 1; inb < nb; inb++){
				for (k = 0; k < kb; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < im; i++){
							tr3db[k][j][i] = trb[inb][k][j][i];	
							tr3d[k][j][i] = tr[inb][k][j][i];	
							vf[k][j][i] = 0;
						}
					}
				}
			

				if (ABS(tracer_flag) == 1){
					//advt2_tr2(tr3db, tr3d, vf);
				}else if (ABS(tracer_flag) == 2){
					//advtt2_tr3(tr3db, tr3d, vf);
				}

				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						rdisp2d[j][i] = 0;	
					}
				}

				//proft(vf, rdisp2d, rdisp2d, 1);
				//bcond(7);
				//exchange3d_mpi(vf, im, jm, kbm1);

				for (k = 0; k < kb; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < im; i++){
							tr[inb][k][j][i] = 
									tr[inb][k][j][i]
								   +0.5f*smoth*(vf[k][j][i]
											   +trb[inb][k][j][i]
											   -2.f*tr[inb][k][j][i]);
						}
					}
				}
			}
			*/
			printf("tracer_flag feature is not supported now!\n");
		}

		
		//modify -uf  //advx changed in advct
		//advu_(uf,w,u,advx,dt,v,egf,egb,e_atmos,drhox,etb,ub,etf);
		
		//advu_gpu(d_ub, d_u, d_uf, d_v, d_w, 
		//		 d_egb, d_egf, d_etb, d_etf,
		//		 d_advx, d_drhox, d_e_atmos, d_dt);
		
		//advu_gpu();

		//modify -vf
		//advv_(vf,w,u,advy,dt,v,egf,egb,e_atmos,drhoy,etb,vb,etf);
		
		//advv_gpu(d_vb, d_v, d_vf, d_u, d_w, 
		//	     d_egb, d_egf, d_etb, d_etf,
		//		 d_advy, d_drhoy, d_e_atmos, d_dt);
		
		//advv_gpu();

		advuv_fusion_gpu();

		//this function ends with a MPI for wubot
		//so we need not copy_back wubot for it is already in Host
		//modify +uf -wubot
		//profu_(etf,km,wusurf,uf,vb,ub,wubot);
		
		//profu_gpu(d_ub, d_uf, d_wusurf, d_wubot, 
		//		  d_vb, d_km, d_etf);
		
		//profu_gpu();

		//modify +vf -wvbot
		//profv_(etf,km,wvsurf,vf,ub,vb,wvbot);
		//in profv, MPI is not included, which is different from profu,
		//I do not know why
		//profv_(etf,km,wvsurf,vf,ub,vb,wvbot);//modify +vf -wvbot
		
		//profv_gpu(d_vb, d_vf, d_wvsurf, d_wvbot, 
		//		  d_ub, d_km, d_etf);
		
		//profv_gpu();
		//profuv_fusion_gpu();
		//bcond_gpu(3);//modify +uf +vf	

		profuv_fusion_overlap_bcond();

		//bcond_gpu(3);//modify +uf +vf	
		bcond_overlap(3, stream[1]);//modify +uf +vf	

		////exchange3d_mpi_gpu(d_uf,im,jm,kbm1);
		////exchange3d_mpi_gpu(d_vf,im,jm,kbm1);

		//exchange3d_cuda_aware_mpi(d_uf,im,jm,kbm1);
		//exchange3d_cuda_aware_mpi(d_vf,im,jm,kbm1);

		//exchange3d_mpi_gpu(d_uf,im,jm,kbm1);
		//exchange3d_mpi_gpu(d_vf,im,jm,kbm1);

		exchange3d_cudaUVA(d_uf, d_uf_east, d_uf_west,
						   d_uf_south, d_uf_north,
						   stream[1], im,jm,kbm1);

		exchange3d_cudaUVA(d_vf, d_vf_east, d_vf_west,
						   d_vf_south, d_vf_north,
						   stream[1], im,jm,kbm1);

		//MPI_Barrier(pom_comm);
		//exchange3d_cuda_ipc(d_uf, d_uf_east, d_uf_west,
		//					stream[1], im,jm,kbm1);
		//exchange3d_cuda_ipc(d_vf, d_vf_east, d_vf_west,
		//				    stream[1], im,jm,kbm1);

		//checkCudaErrors(cudaStreamSynchronize(stream[1]));
		//MPI_Barrier(pom_comm);
		checkCudaErrors(cudaStreamSynchronize(stream[0]));


		//modify +ub, +u, +vb, +v
		
		//mode_internal_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
		//		d_ub, d_u, d_uf, d_vb, d_v, d_vf, d_tps_mode_internal,
		//		d_dz, smoth, kb, jm, im);
#ifdef D3_BLOCK	
		mode_internal_gpu_kernel_3<<<blockPerGrid_D3, threadPerBlock_D3>>>(
				d_ub, d_u, d_uf, d_vb, d_v, d_vf, //d_tps,
				d_dz, smoth, kb, jm, im);
#endif
		mode_internal_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
				d_ub, d_u, d_uf, d_vb, d_v, d_vf, d_tps,
				d_dz, smoth, kb, jm, im);
		
	}

	
	//modify +et, -egb, -etb, -dt, -utb, -vtb, -vfluxb
	mode_internal_gpu_kernel_4<<<blockPerGrid, threadPerBlock>>>(
			d_egb, d_egf, d_etb, d_et, d_etf, d_utb, d_utf, 
			d_vtb, d_vtf, d_vfluxb, d_vfluxf, d_dt, d_h, jm, im);
	
	
	
	/*
	//kernel_0
	checkCudaErrors(cudaMemcpy(u, d_u, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(v, d_v, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	
	//vertvl&&bcond&&exchange3d_mpi
	checkCudaErrors(cudaMemcpy(w, d_w, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	
	//advq
	checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));

	//profq
	//---------------------------------------------------------
	checkCudaErrors(cudaMemcpy(km, d_km, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(kh, d_kh, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(kq, d_kq, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(q2b, d_q2b, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(q2lb, d_q2lb, kb*jm*im*sizeof(float), 
							cudaMemcpyDeviceToHost));
	//---------------------------------------------------------
	//kernel_1
	checkCudaErrors(cudaMemcpy(q2, d_q2, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(q2l, d_q2l, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));

	//advt1
	checkCudaErrors(cudaMemcpy(t, d_t, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(s, d_s, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(tb, d_tb, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(sb, d_sb, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

	//dens
	checkCudaErrors(cudaMemcpy(rho, d_rho, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));

	//bcond(3)
	checkCudaErrors(cudaMemcpy(wubot, d_wubot, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	//kernel_3
	checkCudaErrors(cudaMemcpy(ub, d_ub, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vb, d_vb, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	//kernel_3
	checkCudaErrors(cudaMemcpy(egb, d_egb, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(etb, d_etb, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(et, d_et, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(dt, d_dt, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(utb, d_utb, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vtb, d_vtb, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vfluxb, d_vfluxb, jm*im*sizeof(float), 
			cudaMemcpyDeviceToHost));
	*/
	

//! calculate real w as wr
/*
      call realvertvl(dt,et,w,u,v,etf,etb)
!void realvertvl_(float dt[][i_size], float et[][i_size],
!float w[][j_size][i_size],float u[][j_size][i_size],
!float v[][j_size][i_size],float etf[][i_size],
!float etb[][i_size]);
*/

    //realvertvl_(dt,et,w,u,v,etf,etb);

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_mode_internal);
		mode_internal_time += time_consumed(&start_mode_internal, 
									        &end_mode_internal);
#endif

	return;
}

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
						int mode, int iint, int iprint,
						int kb, int jm, int im){

	int j = blockDim.y*blockIdx.y+threadIdx.y;
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	int k;

	if (j < jm && i < im){

		if (iint % iprint == 1){
			uab_mean[j_off+i] = 0;
			vab_mean[j_off+i] = 0;
			elb_mean[j_off+i] = 0;

			wusurf_mean[j_off+i] = 0;
			wvsurf_mean[j_off+i] = 0;
			wtsurf_mean[j_off+i] = 0;
			wssurf_mean[j_off+i] = 0;
		}
		if (mode == 2){
			uab_mean[j_off+i] += cor[j_off+i];
			vab_mean[j_off+i] += aam2d[j_off+i];
			elb_mean[j_off+i] += e_atmos[j_off+i];
		}else{
			uab_mean[j_off+i] += uab[j_off+i];
			vab_mean[j_off+i] += vab[j_off+i];
			elb_mean[j_off+i] += elb[j_off+i];
		}

		wusurf_mean[j_off+i] += wusurf[j_off+i];	
		wvsurf_mean[j_off+i] += wvsurf[j_off+i];
		wtsurf_mean[j_off+i] += wtsurf[j_off+i];
		wssurf_mean[j_off+i] += wssurf[j_off+i];

		u[kb_1_off+j_off+i] = wubot[j_off+i];
		v[kb_1_off+j_off+i] = wvbot[j_off+i];
		w[kb_1_off+j_off+i] = ustks[j_off+i];
		aam[kb_1_off+j_off+i] = cbc[j_off+i];

		if (iint % iprint == 1){
			for (k = 0; k < kb; k++){
				u_mean[k_off+j_off+i] = 0;
				v_mean[k_off+j_off+i] = 0;
				w_mean[k_off+j_off+i] = 0;
				t_mean[k_off+j_off+i] = 0;
				s_mean[k_off+j_off+i] = 0;
				rho_mean[k_off+j_off+i] = 0;
				kh_mean[k_off+j_off+i] = 0;
				km_mean[k_off+j_off+i] = 0;

				u_mean[k_off+j_off+i] = u_mean[k_off+j_off+i]
									   +u[k_off+j_off+i]
									   +ustks[k_off+j_off+i];

				v_mean[k_off+j_off+i] = v_mean[k_off+j_off+i]
									   +v[k_off+j_off+i]
									   +vstks[k_off+j_off+i];

				w_mean[k_off+j_off+i] += w[k_off+j_off+i];
				t_mean[k_off+j_off+i] += t[k_off+j_off+i];
				s_mean[k_off+j_off+i] += s[k_off+j_off+i];
				rho_mean[k_off+j_off+i] += rho[k_off+j_off+i];
				kh_mean[k_off+j_off+i] += kh[k_off+j_off+i];
				km_mean[k_off+j_off+i] += aam[k_off+j_off+i];
			}
		}else{
			for (k = 0; k < kb; k++){
				u_mean[k_off+j_off+i] = u_mean[k_off+j_off+i]
									   +u[k_off+j_off+i]
									   +ustks[k_off+j_off+i];

				v_mean[k_off+j_off+i] = v_mean[k_off+j_off+i]
									   +v[k_off+j_off+i]
									   +vstks[k_off+j_off+i];

				w_mean[k_off+j_off+i] += w[k_off+j_off+i];
				t_mean[k_off+j_off+i] += t[k_off+j_off+i];
				s_mean[k_off+j_off+i] += s[k_off+j_off+i];
				rho_mean[k_off+j_off+i] += rho[k_off+j_off+i];
				kh_mean[k_off+j_off+i] += kh[k_off+j_off+i];
				km_mean[k_off+j_off+i] += aam[k_off+j_off+i];
			}
		}
	}
}

void store_mean_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_store_mean,
				   end_store_mean;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_store_mean);

#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
					  (j_size+block_j_2D-1)/block_j_2D);

	store_mean_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
		d_uab_mean, d_vab_mean, d_elb_mean, 
		d_wusurf_mean, d_wvsurf_mean, d_wtsurf_mean, d_wssurf_mean,
		d_u, d_v, d_w, d_aam, d_u_mean, d_v_mean, d_w_mean,
		d_t_mean, d_s_mean, d_rho_mean, d_kh_mean, d_km_mean,
		d_cor, d_aam2d, d_e_atmos, d_uab, d_vab, d_elb,
		d_wusurf, d_wvsurf, d_wtsurf, d_wssurf, d_wubot, d_wvbot,
		d_ustks, d_cbc, d_vstks, d_t, d_s, d_rho, d_kh,
		mode, iint, iprint, kb, jm, im);

	num += 1;

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_store_mean);
		store_mean_time += time_consumed(&start_store_mean, 
									     &end_store_mean);
#endif
}


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
						const int mode, const int calc_wind, const int calc_vort,
						const int kb, const int jm, const int im){

	const int j = blockDim.y*blockIdx.y+threadIdx.y;
	const int i = blockDim.x*blockIdx.x+threadIdx.x;
	int k;

	if (j < jm && i < im){
		if (mode == 2){
			usrf_mean[j_off+i] += uab[j_off+i];
			vsrf_mean[j_off+i] += vab[j_off+i];
		}else{
			usrf_mean[j_off+i] += u[j_off+i];	
			vsrf_mean[j_off+i] += v[j_off+i];
		}
	
		elsrf_mean[j_off+i] += elb[j_off+i];	

		if (calc_wind){
			uwsrf_mean[j_off+i] += uwsrf[j_off+i];
			vwsrf_mean[j_off+i] += vwsrf[j_off+i];
		}else{
			uwsrf_mean[j_off+i]	+= wusurf[j_off+i];
			vwsrf_mean[j_off+i] += wvsurf[j_off+i];
		}

		utf_mean[j_off+i] += utf[j_off+i];
		vtf_mean[j_off+i] += vtf[j_off+i];

		for (k = 0; k < kb; k++){
			xstks_mean[k_off+j_off+i] += xstks[k_off+j_off+i];	
			ystks_mean[k_off+j_off+i] += ystks[k_off+j_off+i];
		}

		if (calc_vort){
			celg_mean[j_off+i] += celg[j_off+i];
			ctsurf_mean[j_off+i] += ctsurf[j_off+i];
			ctbot_mean[j_off+i] += ctbot[j_off+i];
			cpvf_mean[j_off+i] += cpvf[j_off+i];
			cjbar_mean[j_off+i] += cjbar[j_off+i];
			cadv_mean[j_off+i] += cadv[j_off+i];
			cten_mean[j_off+i] += cten[j_off+i];
		}
	}
}

void store_surf_mean_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_store_surf_mean,
				   end_store_surf_mean;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_store_surf_mean);

#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
					  (j_size+block_j_2D-1)/block_j_2D);

	if (n_east == -1 && n_north == -1){
		checkCudaErrors(cudaMemcpy(d_elb+(jm-1)*im+(im-1), 
								   d_cor+(jm-1)*im+(im-1),
								   sizeof(float),
								   cudaMemcpyDeviceToDevice));
	}

	store_surf_mean_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
		d_usrf_mean, d_vsrf_mean, d_elb, d_elsrf_mean,
		d_uwsrf_mean, d_vwsrf_mean, d_utf_mean, d_vtf_mean,
		d_xstks_mean, d_ystks_mean, d_celg_mean, d_ctsurf_mean,
		d_ctbot_mean, d_cpvf_mean, d_cjbar_mean,
		d_cadv_mean, d_cten_mean,

		d_uab, d_vab, d_u, d_v, d_cor, d_uwsrf, d_vwsrf, 
		d_wusurf, d_wvsurf, d_utf, d_vtf, d_xstks, d_ystks,
		d_celg, d_ctsurf, d_ctbot, d_cpvf, d_cjbar, d_cadv,
		d_cten, 
		mode, calc_wind, calc_vort, kb, jm, im);

	nums += 1;

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_store_surf_mean);
		store_surf_mean_time += time_consumed(&start_store_surf_mean, 
									          &end_store_surf_mean);
#endif
}



//the same with HOST print_section function
void print_section_gpu(){

    int i,j,k;
    float area_tot,vol_tot,d_area,d_vol;
    float elev_ave, temp_ave, salt_ave;
	float u_ave, v_ave;

	//if (iint%iprint == 0){
	if (iint == iend){
		checkCudaErrors(cudaMemcpy(tb, d_tb, kb*jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(sb, d_sb, kb*jm*im*sizeof(float), 
						cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(dt, d_dt, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(et, d_et, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(ub, d_ub, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vb, d_vb, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		if (my_task == master_task){
			printf("*******************************************************\n");
			printf("time = %f, iint = %d, iext = %d, iprint = %d\n", 
					model_time, iint, iext, iprint);
		}
		sum0i_mpi(&error_status, master_task);
		bcast0d_mpi(&error_status, master_task);

		if (error_status != 0){
			printf("POM terminated with error! in file: %s, function: %s, line: %d\n",
					__FILE__, __func__, __LINE__);
			//finalize_mpi();
		}
		
		vol_tot = 0.0f;
		area_tot = 0.0f;
		temp_ave = 0.0f;
		salt_ave = 0.0f;
		elev_ave = 0.0f;
		u_ave = 0.0f;
		v_ave = 0.0f;

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					d_area = dx[j][i]*dy[j][i];
					d_vol = d_area*dt[j][i]*dz[k]*fsm[j][i];
					vol_tot = vol_tot + d_vol;
					temp_ave = temp_ave + tb[k][j][i]*d_vol;
					salt_ave = salt_ave + sb[k][j][i]*d_vol;
					u_ave = u_ave + ub[k][j][i]*d_vol;
					v_ave = v_ave + vb[k][j][i]*d_vol;
				}
			}
		} 

		//exchange2d_mpi(et,im,jm);

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				d_area = dx[j][i]*dy[j][i];
				area_tot += d_area;
				elev_ave += et[j][i]*d_area;
			}
		}

		if (my_task == master_task){
			printf("****************************************************\n");
			printf("et = %+30.24e m, tb = %+30.24e deg, sb = %+30.24e psu\n",
					elev_ave/area_tot, temp_ave/vol_tot +tbias, 
					salt_ave/vol_tot+sbias);	
			printf("ub = %+30.24e m/s, vb = %+30.24e m/s\n",
					u_ave/vol_tot, v_ave/vol_tot);	

			printf("****************************************************\n");
		}

		sum0f_mpi(&temp_ave, master_task);
		sum0f_mpi(&salt_ave, master_task);
		sum0f_mpi(&elev_ave, master_task);
		sum0f_mpi(&vol_tot, master_task);
		sum0f_mpi(&area_tot, master_task);


		temp_ave /= vol_tot;
		salt_ave /= vol_tot;
		u_ave /= vol_tot;
		v_ave /= vol_tot;
		elev_ave /= area_tot;


		if (my_task == master_task){
			printf("****************************************************\n");
			printf("et = %+30.24e m, tb = %+30.24e deg, sb = %+30.24e psu\n",
					elev_ave, temp_ave+tbias, salt_ave+sbias);	
			printf("ub = %+30.24e m/s, vb = %+30.24e m/s\n",
					u_ave, v_ave);	

			printf("****************************************************\n");
//#ifndef TIME_DISABLE
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_advance",
					advance_time/1.E+6, 100.f*advance_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_momentum3d",
					momentum3d_time/1.E+6, 100.f*momentum3d_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_mode_interaction",
					mode_interaction_time/1.E+6, 100.f*mode_interaction_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_mode_external",
					mode_external_time/1.E+6, 100.f*mode_external_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_mode_internal",
					mode_internal_time/1.E+6, 100.f*mode_internal_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_store_mean",
					store_mean_time/1.E+6, 100.f*store_mean_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_store_surf_mean",
					store_surf_mean_time/1.E+6, 100.f*store_surf_mean_time/advance_time);

			printf("*******************************************************\n");

			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_vort",
					vort_time/1.E+6, 100.f*vort_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advave",
					advave_time/1.E+6, 100.f*advave_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advct",
					advct_time/1.E+6, 100.f*advct_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advq",
					advq_time/1.E+6, 100.f*advq_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_fusion_advq",
					advq_fusion_time/1.E+6, 100.f*advq_fusion_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advt1",
					advt1_time/1.E+6, 100.f*advt1_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advt2",
					advt2_time/1.E+6, 100.f*advt2_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advu",
					advu_time/1.E+6, 100.f*advu_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advv",
					advv_time/1.E+6, 100.f*advv_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_fusion_advuv",
					advuv_fusion_time/1.E+6, 100.f*advuv_fusion_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_baropg",
					baropg_time/1.E+6, 100.f*baropg_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_dens",
					dens_time/1.E+6, 100.f*dens_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_profq",
					profq_time/1.E+6, 100.f*profq_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_proft",
					proft_time/1.E+6, 100.f*proft_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_fusion_proft",
					proft_fusion_time/1.E+6, 100.f*proft_fusion_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_profu",
					profu_time/1.E+6, 100.f*profu_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_profv",
					profv_time/1.E+6, 100.f*profv_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_fusion_profuv",
					profuv_fusion_time/1.E+6, 100.f*profuv_fusion_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_smol_adif",
					smol_adif_time/1.E+6, 100.f*smol_adif_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_vertvl",
					vertvl_time/1.E+6, 100.f*vertvl_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_realvertvl",
					realvertvl_time/1.E+6, 100.f*realvertvl_time/advance_time);

			printf("*******************************************************\n");

			printf("%-30s:  %7.3lf; %7.3f%\n", "time_3d_mpi",
					exchange3d_mpi_time/1.E+6, 100.f*exchange3d_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_2d_mpi",
					exchange2d_mpi_time/1.E+6, 100.f*exchange2d_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_3d_cuda_aware_mpi",
					exchange3d_cuda_aware_mpi_time/1.E+6, 100.f*exchange3d_cuda_aware_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_3d_cuda_ipc",
					exchange3d_cuda_ipc_time/1.E+6, 100.f*exchange3d_cuda_ipc_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_2d_cuda_aware_mpi",
					exchange2d_cuda_aware_mpi_time/1.E+6, 100.f*exchange2d_cuda_aware_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_2d_cuda_ipc",
					exchange2d_cuda_ipc_time/1.E+6, 100.f*exchange2d_cuda_ipc_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_xperi2d_mpi",
					xperi2d_mpi_time/1.E+6, 100.f*xperi2d_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_xperi2d_cuda_aware_mpi",
					xperi2d_cuda_aware_mpi_time/1.E+6, 100.f*xperi2d_cuda_aware_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_yperi2d_mpi",
					yperi2d_mpi_time/1.E+6, 100.f*yperi2d_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_xperi3d_mpi",
					xperi3d_mpi_time/1.E+6, 100.f*xperi3d_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_xperi3d_cuda_aware_mpi",
					xperi3d_cuda_aware_mpi_time/1.E+6, 100.f*xperi3d_cuda_aware_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_yperi3d_mpi",
					yperi3d_mpi_time/1.E+6, 100.f*yperi3d_mpi_time/advance_time);
			printf("*******************************************************\n");

			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond",
					bcond_time/1.E+6, 100.f*bcond_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond_1",
					bcond_time_1/1.E+6, 100.f*bcond_time_1/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond_2",
					bcond_time_2/1.E+6, 100.f*bcond_time_2/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond_3",
					bcond_time_3/1.E+6, 100.f*bcond_time_3/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond_4",
					bcond_time_4/1.E+6, 100.f*bcond_time_4/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond_5",
					bcond_time_5/1.E+6, 100.f*bcond_time_5/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_bcond_6",
					bcond_time_6/1.E+6, 100.f*bcond_time_6/advance_time);
			printf("*******************************************************\n\n\n");
//#endif
		}

	}
/*
      call call_time_end(print_section_time_end_xsz)
      print_section_time_xsz = print_section_time_end_xsz -
     $                         print_section_time_start_xsz +
     $                         print_section_time_xsz

      time_total_xsz = print_section_time_end_xsz -
     $                 time_start_xsz
*/

	return;
}


//the same with HOST check velocity function
void check_velocity_gpu(){

    //float vamax,atot,darea,dvol,eaver,saver,taver,vtot,tsalt;
	float vamax;
	//int i,j,k;
	int i,j;
	int imax, jmax;
	vamax=0.0f;


	checkCudaErrors(cudaMemcpy(vaf, d_vaf, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
/*
      do j=1,jm
        do i=1,im
          if(abs(vaf(i,j)).ge.vamax) then
            vamax=abs(vaf(i,j))
            imax=i
            jmax=j
          end if
        end do
      end do
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			if (ABS(vaf[j][i]) >= vamax){
				vamax = ABS(vaf[j][i]);	
				imax = i;
				jmax = j;
			}
		}
	}

/*
      if(vamax.gt.vmaxl) then
        if(my_task.eq.master_task.and.error_status.eq.0) write(6,'(/
     $    ''Error: velocity condition violated''/''time ='',f9.4,
     $    '', iint ='',i8,'', iext ='',i8,'', iprint ='',i8,/
     $    ''vamax ='',e12.3,''   imax,jmax ='',2i5)')
     $    time,iint,iext,iprint,vamax,imax,jmax
        error_status=1
      end if
*/
	if (vamax > vmaxl){
		if (my_task == master_task && error_status == 0){
			printf("Error: velocity condition violated! time = %f, iint = %d, \
					iext = %d, iprint = %d, vamax = %f, imax = %d, jmax = %d\n",
					//time, iint, iext, iprint, vamax, imax, jmax);	
					model_time, iint, iext, iprint, vamax, imax, jmax);	
			error_status = 1;
		}
	}

/*
      call call_time_end(check_velocity_time_end_xsz)
      check_velocity_time_xsz = check_velocity_time_end_xsz - 
     $                          check_velocity_time_start_xsz +
     $                          check_velocity_time_xsz

*/
    return;
}

void output_copy_back(int num){

	if (num == 0){
		checkCudaErrors(cudaMemcpy(uab, d_uab, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vab, d_vab, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(elb, d_elb, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		//xsz_test check
		checkCudaErrors(cudaMemcpy(et, d_et, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(u, d_u, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(v, d_v, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(w, d_w, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(t, d_t, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(s, d_s, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(rho, d_rho, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(kh, d_kh, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(km, d_km, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	}else{
		checkCudaErrors(cudaMemcpy(uab_mean, d_uab_mean, 
								   jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vab_mean, d_vab_mean, 
								   jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(elb_mean, d_elb_mean, 
								   jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(u_mean, d_u_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(v_mean, d_v_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(w_mean, d_w_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(t_mean, d_t_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(s_mean, d_s_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(rho_mean, d_rho_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(kh_mean, d_kh_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(km_mean, d_km_mean, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void advance_gpu(){
    //timer_now_(time, ramp, iprint, iint);

	struct timeval start_advance,
				   end_advance;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advance);

    get_time_gpu();

	//////////////////////////////////////////////////////////////////
    surface_forcing_gpu();

    //lateral_viscosity(advx,advy,rho,drhox,drhoy,aam,v,u,dt,ub,vb,ramp);
	momentum3d_gpu(); 

    mode_interaction_gpu();

    for (iext = 1; iext <= isplit; iext ++){
	//	mode_external(elf, advua, advva, fluxua, fluxva, uaf, vaf, etf, ua, va, el, elb, d, uab, vab, egf, utf, vtf,
    //    
	//		vfluxf, wusurf, aam2d, wubot, wvbot, adx2d, drx2d, ady2d, wvsurf,ramp, dry2d, iext, e_atmos);
	
		mode_external_gpu();
	}

	////////////////////////////////////////////////////////////////////
    //mode_internal(u,v,w,uf,vf,km,kh,q2b,q2lb,q2l,q2,t,tb,s,sb,wubot,wvbot,egb,etb,et,dt,utb,vtb,vfluxb,vfluxf,
    // 
    //           iint,aam,rho,advx,advy,ub,vb,utf,vtf,etf,wusurf,wvsurf,egf,e_atmos,drhox,drhoy,wtsurf,wssurf,swrad);
	
    mode_internal_gpu();

    print_section_gpu();

//	check_nan();

	store_mean_gpu();

	store_surf_mean_gpu();

//	write_output();
//
//	if (iint % irestart == 0)
//		write_restart_pnetcdf();

    //check_velocity_gpu();
	
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&end_advance);
	advance_time += time_consumed(&start_advance, 
								  &end_advance);

}


