#include<stdio.h>

#include"csolver_gpu.h"
#include"csolver_gpu_kernel.h"
#include"cparallel_mpi_gpu.h"

#include"cu_data.h"

#include"data.h"
#include"timer_all.h"
#include"cparallel_mpi.h"


__global__ void
advct_gpu_kernel_0(float * __restrict__ curv, 
				   float * __restrict__ xflux_advx,
				   float * __restrict__ xflux_advy,
				   float * __restrict__ yflux_advx,
				   float * __restrict__ yflux_advy,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ ub,
				   const float * __restrict__ vb,
				   const float * __restrict__ dt,
				   const float * __restrict__ aam,
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy,
				   int kb, int jm, int im){

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

#ifdef D3_BLOCK
	/*
	if (k < kb && j < jm && i < im){
		curv[k_off+j_off+i] = 0;
		advx[k_off+j_off+i] = 0;
		xflux[k_off+j_off+i] = 0;
		yflux[k_off+j_off+i] = 0;
	}
	*/

	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]+v[k_off+j_off+i])*(dy[j_off+(i+1)]-dy[j_off+(i-1)]) - 
									 (u[k_off+j_off+(i+1)]+u[k_off+j_off+i])*(dx[j_A1_off+i]-dx[j_1_off+i])) /
									(dx[j_off+i]*dy[j_off+i]);
	}

#else
	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				curv[k][j][i] = 0;
				advx[k][j][i] = 0;
				xflux[k][j][i] = 0;
				yflux[k][j][i] = 0;
			}
		}
	}
	*/ 

	/*
	if (k < kb && j < jm && i < im){
		curv[k_off+j_off+i] = 0;
		advx[k_off+j_off+i] = 0;
		xflux[k_off+j_off+i] = 0;
		yflux[k_off+j_off+i] = 0;
	}
	*/

	/*
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			curv[k_off+j_off+i] = 0;
			advx[k_off+j_off+i] = 0;
			xflux[k_off+j_off+i] = 0;
			yflux[k_off+j_off+i] = 0;
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				curv[k][j][i] = 0.25f*((v[k][j+1][i]+v[k][j][i])*(dy[j][i+1]-dy[j][i-1]) - (u[k][j][i+1]+u[k][j][i])*(dx[j+1][i]-dx[j-1][i]))/(dx[j][i]*dy[j][i]);
			}
		}
	}
	*/

	/*
	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]+v[k_off+j_off+i])*(dy[j_off+(i+1)]-dy[j_off+(i-1)]) - 
									 (u[k_off+j_off+(i+1)]+u[k_off+j_off+i])*(dx[j_A1_off+i]-dx[j_1_off+i])) /
									(dx[j_off+i]*dy[j_off+i]);
	}
	*/

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 0; k < kbm1; k++){
			curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]
										    +v[k_off+j_off+i])
										 *(dy[j_off+(i+1)]
											-dy[j_off+(i-1)]) 
										-(u[k_off+j_off+(i+1)]
											+u[k_off+j_off+i])
										 *(dx[j_A1_off+i]-dx[j_1_off+i])) 
										/(dx[j_off+i]*dy[j_off+i]);
		}
	}

	if (blockIdx.x > 0 && blockIdx.x < gridDim.x-1 &&
		blockIdx.y > 0 && blockIdx.y < gridDim.y-1){

		for (k = 0; k < kbm1; k++){
			//float dt_j_i = dt[j_off+i];
			//float dt_j_i_1 = dt[j_off+i-1];
			//float dt_j_i_A1 = dt[j_off+i+1];
			//float dt_j_1_i = dt[j_1_off+i];
			//float dt_j_A1_i = dt[j_A1_off+i];
			//float dt_j_1_i_1 = dt[j_1_off+i-1];

			//float dx_j_i = dx[j_off+i];
			//float dx_j_i_1 = dx[j_off+i-1];
			//float dx_j_1_i = dx[j_1_off+i];
			//float dx_j_1_i_1 = dx[j_1_off+i-1];

			//float dy_j_i = dy[j_off+i];
			//float dy_j_i_1 = dy[j_off+i-1];
			//float dy_j_1_i = dy[j_1_off+i];
			//float dy_j_1_i_1 = dy[j_1_off+i-1];

			float xflux_advx_tmp, yflux_advx_tmp;
			float xflux_advy_tmp, yflux_advy_tmp;
			float dtaam;

			dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
						    +dt[j_1_off+i]+dt[j_1_off+(i-1)])	
						 *(aam[k_off+j_off+i]
							+aam[k_off+j_off+(i-1)] 
						    +aam[k_off+j_1_off+i]
							+aam[k_off+j_1_off+(i-1)]);
			/*
			dtaam = 0.25f*(dt_j_i+dt_j_i_1+dt_j_1_i+dt_j_1_i_1)	
						 *(aam[k_off+j_off+i]
							+aam[k_off+j_off+(i-1)] 
						    +aam[k_off+j_1_off+i]
							+aam[k_off+j_1_off+(i-1)]);

			*/
			xflux_advx_tmp = 0.125f*((dt[j_off+(i+1)]
				          			   +dt[j_off+i])
				          			  *u[k_off+j_off+(i+1)] 
				          		    +(dt[j_off+i]
				          			   +dt[j_off+(i-1)])
				          			  *u[k_off+j_off+i]) 
				          		   *(u[k_off+j_off+(i+1)]
				          			 +u[k_off+j_off+i]);
			/*
			xflux_advx_tmp = 0.125f*((dt_j_i_A1+dt_j_i)
				          			  *u[k_off+j_off+(i+1)] 
				          		    +(dt_j_i+dt_j_i_1)
				          			  *u[k_off+j_off+i]) 
				          		   *(u[k_off+j_off+(i+1)]
				          			 +u[k_off+j_off+i]);
			*/

			xflux_advy_tmp = 0.125f*((dt[j_off+i]
										+dt[j_off+(i-1)])
									  *u[k_off+j_off+i] 
									+(dt[j_1_off+i]
										+dt[j_1_off+(i-1)])
									  *u[k_off+j_1_off+i]) 
								   *(v[k_off+j_off+i]
									  +v[k_off+j_off+(i-1)]);	
			/*
			xflux_advy_tmp = 0.125f*((dt_j_i+dt_j_i_1)
									  *u[k_off+j_off+i] 
									+(dt_j_1_i+dt_j_1_i_1)
									  *u[k_off+j_1_off+i]) 
								   *(v[k_off+j_off+i]
									  +v[k_off+j_off+(i-1)]);	
			*/
		

			yflux_advx_tmp = 0.125f*((dt[j_off+i]
				          			  +dt[j_1_off+i])
				          			*v[k_off+j_off+i] 
				          		   +(dt[j_off+(i-1)]
				          			  +dt[j_1_off+(i-1)])
				          			*v[k_off+j_off+(i-1)]) 
				          		 *(u[k_off+j_off+i]
				          		   +u[k_off+j_1_off+i]);
			/*
			yflux_advx_tmp = 0.125f*((dt_j_i+dt_j_1_i)
				          			*v[k_off+j_off+i] 
				          		   +(dt_j_i_1+dt_j_1_i_1)
				          			*v[k_off+j_off+(i-1)]) 
				          		 *(u[k_off+j_off+i]
				          		   +u[k_off+j_1_off+i]);
			*/
			

			yflux_advy_tmp = 0.125f*((dt[j_A1_off+i]
				          			  +dt[j_off+i])
				          			 *v[k_off+j_A1_off+i] 
				          		   +(dt[j_off+i]
				          			  +dt[j_1_off+i])
				          			 *v[k_off+j_off+i]) 
				          		  *(v[k_off+j_A1_off+i] 
				          			 +v[k_off+j_off+i]);		
			/*
			yflux_advy_tmp = 0.125f*((dt_j_A1_i+dt_j_i)
				          			 *v[k_off+j_A1_off+i] 
				          		   +(dt_j_i+dt_j_1_i)
				          			 *v[k_off+j_off+i]) 
				          		  *(v[k_off+j_A1_off+i] 
				          			 +v[k_off+j_off+i]);		
			*/
			
			xflux_advx_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				             *2.0f*(ub[k_off+j_off+(i+1)]
				     	     	 -ub[k_off+j_off+i])/dx[j_off+i];
			/*
			xflux_advx_tmp -= dt_j_i*aam[k_off+j_off+i]
				             *2.0f*(ub[k_off+j_off+(i+1)]
				     	     	 -ub[k_off+j_off+i])/dx_j_i;
			*/

			xflux_advx_tmp *= dy[j_off+i];

			/*
			xflux_advx_tmp *= dy_j_i;
			*/

			
			yflux_advy_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				        *2.0f*(vb[k_off+j_A1_off+i]
				     		   -vb[k_off+j_off+i])
				        /dy[j_off+i];
			/*
			yflux_advy_tmp -= dt_j_i*aam[k_off+j_off+i]
								*2.0f*(vb[k_off+j_A1_off+i]
				     				   -vb[k_off+j_off+i])
				        		/dy_j_i;
			*/

			yflux_advy_tmp *= dx[j_off+i];
			/*
			yflux_advy_tmp *= dx_j_i;
			*/

			
			xflux_advy_tmp -= dtaam
				     	     *((ub[k_off+j_off+i]
				     	         -ub[k_off+j_1_off+i])
				     	       /(dy[j_off+i]+dy[j_off+(i-1)]
				     	        +dy[j_1_off+i]+dy[j_1_off+(i-1)]) 
				     	      +(vb[k_off+j_off+i]
				     	     	-vb[k_off+j_off+(i-1)])
				     	       /(dx[j_off+i]+dx[j_off+(i-1)]
				     	        +dx[j_1_off+i]+dx[j_1_off+(i-1)]));

			/*
			xflux_advy_tmp -= dtaam
				     	     *((ub[k_off+j_off+i]
				     	         -ub[k_off+j_1_off+i])
				     	       /(dy_j_i+dy_j_i_1+dy_j_1_i+dy_j_1_i_1) 
				     	      +(vb[k_off+j_off+i]
				     	     	-vb[k_off+j_off+(i-1)])
				     	       /(dx_j_i+dx_j_i_1+dx_j_1_i+dx_j_1_i_1));
			*/

			xflux_advy_tmp = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]
									+dy[j_1_off+i]+dy[j_1_off+(i-1)])
				     		      *xflux_advy_tmp;

			/*
			xflux_advy_tmp = 0.25f*(dy_j_i+dy_j_i_1+dy_j_1_i+dy_j_1_i_1)
				     		      *xflux_advy_tmp;
			*/

			yflux_advx_tmp -= dtaam
				     	     *((ub[k_off+j_off+i]
				     	          -ub[k_off+j_1_off+i])
				     	        /(dy[j_off+i]
				     	     	   +dy[j_off+(i-1)]
				     	     	   +dy[j_1_off+i]
				     	     	   +dy[j_1_off+(i-1)]) 
				     	      +(vb[k_off+j_off+i]
				     	     	 -vb[k_off+j_off+(i-1)])
				     	        /(dx[j_off+i]
				     	     	   +dx[j_off+(i-1)]
				     	     	   +dx[j_1_off+i]
				     	     	   +dx[j_1_off+(i-1)]));
			/*
			yflux_advx_tmp -= dtaam
				     	     *((ub[k_off+j_off+i]
				     	          -ub[k_off+j_1_off+i])
				     	        /(dy_j_i+dy_j_i_1+dy_j_1_i+dy_j_1_i_1) 
				     	      +(vb[k_off+j_off+i]
				     	     	 -vb[k_off+j_off+(i-1)])
				     	        /(dx_j_i+dx_j_i_1+dx_j_1_i+dx_j_1_i_1));
			*/

			yflux_advx_tmp = 0.25f*(dx[j_off+i]
				     		     	+dx[j_off+(i-1)]
				     		     	+dx[j_1_off+i]
				     		     	+dx[j_1_off+(i-1)]) 
				     		      *yflux_advx_tmp;

			/*
			yflux_advx_tmp = 0.25f*(dx_j_i+dx_j_i_1+dx_j_1_i+dx_j_1_i_1) 
				     		      *yflux_advx_tmp;
			*/
			


			xflux_advx[k_off+j_off+i] = xflux_advx_tmp;
			xflux_advy[k_off+j_off+i] = xflux_advy_tmp;
			yflux_advx[k_off+j_off+i] = yflux_advx_tmp;
			yflux_advy[k_off+j_off+i] = yflux_advy_tmp;
		}

	}else{
		if(j < jm && i < im){
			for (k = 0; k < kbm1; k++){
				float xflux_advx_tmp, yflux_advx_tmp;
				float xflux_advy_tmp, yflux_advy_tmp;
				float dtaam;

				if (i > 0 && j > 0){
					dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
								    +dt[j_1_off+i]+dt[j_1_off+(i-1)])	
								 *(aam[k_off+j_off+i]
									+aam[k_off+j_off+(i-1)] 
								    +aam[k_off+j_1_off+i]
									+aam[k_off+j_1_off+(i-1)]);
				}

				if (i < imm1 && i > 0){
					xflux_advx_tmp = 0.125f*((dt[j_off+(i+1)]
						          			   +dt[j_off+i])
						          			  *u[k_off+j_off+(i+1)] 
						          		    +(dt[j_off+i]
						          			   +dt[j_off+(i-1)])
						          			  *u[k_off+j_off+i]) 
						          		   *(u[k_off+j_off+(i+1)]
						          			 +u[k_off+j_off+i]);
				}

				if (i > 0 && j > 0){
					xflux_advy_tmp = 0.125f*((dt[j_off+i]
												+dt[j_off+(i-1)])
											  *u[k_off+j_off+i] 
											+(dt[j_1_off+i]
												+dt[j_1_off+(i-1)])
											  *u[k_off+j_1_off+i]) 
										   *(v[k_off+j_off+i]
											  +v[k_off+j_off+(i-1)]);	
		
				}

				if (i > 0 && j > 0){
					yflux_advx_tmp = 0.125f*((dt[j_off+i]
						          			  +dt[j_1_off+i])
						          			*v[k_off+j_off+i] 
						          		   +(dt[j_off+(i-1)]
						          			  +dt[j_1_off+(i-1)])
						          			*v[k_off+j_off+(i-1)]) 
						          		 *(u[k_off+j_off+i]
						          		   +u[k_off+j_1_off+i]);
					
				}

				if (j > 0 && j < jmm1){
					yflux_advy_tmp = 0.125f*((dt[j_A1_off+i]
						          			  +dt[j_off+i])
						          			 *v[k_off+j_A1_off+i] 
						          		   +(dt[j_off+i]
						          			  +dt[j_1_off+i])
						          			 *v[k_off+j_off+i]) 
						          		  *(v[k_off+j_A1_off+i] 
						          			 +v[k_off+j_off+i]);		
				}
				

				if (i > 0 && j > 0 && i < imm1){

					xflux_advx_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
						             *2.0f*(ub[k_off+j_off+(i+1)]
						     	     	 -ub[k_off+j_off+i])/dx[j_off+i];

					xflux_advx_tmp *= dy[j_off+i];
				}

				if (j > 0 && i > 0 && j < jmm1){
				
					yflux_advy_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
						        *2.0f*(vb[k_off+j_A1_off+i]
						     		   -vb[k_off+j_off+i])
						        /dy[j_off+i];

					yflux_advy_tmp *= dx[j_off+i];
				}

				if (j > 0 && i > 0 && j < jmm1){
				
					xflux_advy_tmp -= dtaam
						     	     *((ub[k_off+j_off+i]
						     	         -ub[k_off+j_1_off+i])
						     	       /(dy[j_off+i]+dy[j_off+(i-1)]
						     	        +dy[j_1_off+i]+dy[j_1_off+(i-1)]) 
						     	      +(vb[k_off+j_off+i]
						     	     	-vb[k_off+j_off+(i-1)])
						     	       /(dx[j_off+i]+dx[j_off+(i-1)]
						     	        +dx[j_1_off+i]+dx[j_1_off+(i-1)]));

					xflux_advy_tmp = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]
											+dy[j_1_off+i]+dy[j_1_off+(i-1)])
						     		      *xflux_advy_tmp;
				}




				if (i > 0 && j > 0 && i < imm1){
					yflux_advx_tmp -= dtaam
						     	     *((ub[k_off+j_off+i]
						     	          -ub[k_off+j_1_off+i])
						     	        /(dy[j_off+i]
						     	     	   +dy[j_off+(i-1)]
						     	     	   +dy[j_1_off+i]
						     	     	   +dy[j_1_off+(i-1)]) 
						     	      +(vb[k_off+j_off+i]
						     	     	 -vb[k_off+j_off+(i-1)])
						     	        /(dx[j_off+i]
						     	     	   +dx[j_off+(i-1)]
						     	     	   +dx[j_1_off+i]
						     	     	   +dx[j_1_off+(i-1)]));

					yflux_advx_tmp = 0.25f*(dx[j_off+i]
						     		     	+dx[j_off+(i-1)]
						     		     	+dx[j_1_off+i]
						     		     	+dx[j_1_off+(i-1)]) 
						     		      *yflux_advx_tmp;
					
				}
				xflux_advx[k_off+j_off+i] = xflux_advx_tmp;
				xflux_advy[k_off+j_off+i] = xflux_advy_tmp;
				yflux_advx[k_off+j_off+i] = yflux_advx_tmp;
				yflux_advy[k_off+j_off+i] = yflux_advy_tmp;
			}
		}
	}

#endif

}

__global__ void
advct_inner_gpu_kernel_0(float * __restrict__ curv, 
				   float * __restrict__ xflux_advx,
				   float * __restrict__ xflux_advy,
				   float * __restrict__ yflux_advx,
				   float * __restrict__ yflux_advy,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ ub,
				   const float * __restrict__ vb,
				   const float * __restrict__ dt,
				   const float * __restrict__ aam,
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy,
				   int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	//if (j > 32 && i > 32 && j < jm-33 && i < im-33){
	//	for (k = 0; k < kbm1; k++){
	//		curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]
	//									    +v[k_off+j_off+i])
	//									 *(dy[j_off+(i+1)]
	//										-dy[j_off+(i-1)]) 
	//									-(u[k_off+j_off+(i+1)]
	//										+u[k_off+j_off+i])
	//									 *(dx[j_A1_off+i]-dx[j_1_off+i])) 
	//									/(dx[j_off+i]*dy[j_off+i]);
	//	}
	//}

	if (j > 32 && i > 32 && j < jm-33 && i < im-33){

		for (k = 0; k < kbm1; k++){
			float xflux_advx_tmp, yflux_advx_tmp;
			float xflux_advy_tmp, yflux_advy_tmp;
			float dtaam;

			curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]
										    +v[k_off+j_off+i])
										 *(dy[j_off+(i+1)]
											-dy[j_off+(i-1)]) 
										-(u[k_off+j_off+(i+1)]
											+u[k_off+j_off+i])
										 *(dx[j_A1_off+i]-dx[j_1_off+i])) 
										/(dx[j_off+i]*dy[j_off+i]);

			dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
						    +dt[j_1_off+i]+dt[j_1_off+(i-1)])	
						 *(aam[k_off+j_off+i]
							+aam[k_off+j_off+(i-1)] 
						    +aam[k_off+j_1_off+i]
							+aam[k_off+j_1_off+(i-1)]);

			xflux_advx_tmp = 0.125f*((dt[j_off+(i+1)]
				          			   +dt[j_off+i])
				          			  *u[k_off+j_off+(i+1)] 
				          		    +(dt[j_off+i]
				          			   +dt[j_off+(i-1)])
				          			  *u[k_off+j_off+i]) 
				          		   *(u[k_off+j_off+(i+1)]
				          			 +u[k_off+j_off+i]);

			xflux_advy_tmp = 0.125f*((dt[j_off+i]
										+dt[j_off+(i-1)])
									  *u[k_off+j_off+i] 
									+(dt[j_1_off+i]
										+dt[j_1_off+(i-1)])
									  *u[k_off+j_1_off+i]) 
								   *(v[k_off+j_off+i]
									  +v[k_off+j_off+(i-1)]);	

			yflux_advx_tmp = 0.125f*((dt[j_off+i]
				          			  +dt[j_1_off+i])
				          			*v[k_off+j_off+i] 
				          		   +(dt[j_off+(i-1)]
				          			  +dt[j_1_off+(i-1)])
				          			*v[k_off+j_off+(i-1)]) 
				          		 *(u[k_off+j_off+i]
				          		   +u[k_off+j_1_off+i]);

			yflux_advy_tmp = 0.125f*((dt[j_A1_off+i]
				          			  +dt[j_off+i])
				          			 *v[k_off+j_A1_off+i] 
				          		   +(dt[j_off+i]
				          			  +dt[j_1_off+i])
				          			 *v[k_off+j_off+i]) 
				          		  *(v[k_off+j_A1_off+i] 
				          			 +v[k_off+j_off+i]);		
			
			xflux_advx_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				             *2.0f*(ub[k_off+j_off+(i+1)]
				     	     	 -ub[k_off+j_off+i])/dx[j_off+i];

			xflux_advx_tmp *= dy[j_off+i];
			
			yflux_advy_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				        *2.0f*(vb[k_off+j_A1_off+i]
				     		   -vb[k_off+j_off+i])
				        /dy[j_off+i];

			yflux_advy_tmp *= dx[j_off+i];

			
			xflux_advy_tmp -= dtaam
				     	     *((ub[k_off+j_off+i]
				     	         -ub[k_off+j_1_off+i])
				     	       /(dy[j_off+i]+dy[j_off+(i-1)]
				     	        +dy[j_1_off+i]+dy[j_1_off+(i-1)]) 
				     	      +(vb[k_off+j_off+i]
				     	     	-vb[k_off+j_off+(i-1)])
				     	       /(dx[j_off+i]+dx[j_off+(i-1)]
				     	        +dx[j_1_off+i]+dx[j_1_off+(i-1)]));

			xflux_advy_tmp = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]
									+dy[j_1_off+i]+dy[j_1_off+(i-1)])
				     		      *xflux_advy_tmp;

			yflux_advx_tmp -= dtaam
				     	     *((ub[k_off+j_off+i]
				     	          -ub[k_off+j_1_off+i])
				     	        /(dy[j_off+i]
				     	     	   +dy[j_off+(i-1)]
				     	     	   +dy[j_1_off+i]
				     	     	   +dy[j_1_off+(i-1)]) 
				     	      +(vb[k_off+j_off+i]
				     	     	 -vb[k_off+j_off+(i-1)])
				     	        /(dx[j_off+i]
				     	     	   +dx[j_off+(i-1)]
				     	     	   +dx[j_1_off+i]
				     	     	   +dx[j_1_off+(i-1)]));

			yflux_advx_tmp = 0.25f*(dx[j_off+i]
				     		     	+dx[j_off+(i-1)]
				     		     	+dx[j_1_off+i]
				     		     	+dx[j_1_off+(i-1)]) 
				     		      *yflux_advx_tmp;

			xflux_advx[k_off+j_off+i] = xflux_advx_tmp;
			xflux_advy[k_off+j_off+i] = xflux_advy_tmp;
			yflux_advx[k_off+j_off+i] = yflux_advx_tmp;
			yflux_advy[k_off+j_off+i] = yflux_advy_tmp;
		}

	}else{
		if(i < im && j < jm){
			for (k = 0; k < kbm1; k++){
				float yflux_advx_tmp, xflux_advy_tmp;
				float dtaam;

				dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
							    +dt[j_1_off+i]+dt[j_1_off+(i-1)])	
							 *(aam[k_off+j_off+i]
								+aam[k_off+j_off+(i-1)] 
							    +aam[k_off+j_1_off+i]
								+aam[k_off+j_1_off+(i-1)]);


				if (j < jmm1){

					xflux_advy_tmp = 0.125f*((dt[j_off+i]
												+dt[j_off+(i-1)])
											  *u[k_off+j_off+i] 
											+(dt[j_1_off+i]
												+dt[j_1_off+(i-1)])
											  *u[k_off+j_1_off+i]) 
										   *(v[k_off+j_off+i]
											  +v[k_off+j_off+(i-1)]);	
		
				
					xflux_advy_tmp -= dtaam
						     	     *((ub[k_off+j_off+i]
						     	         -ub[k_off+j_1_off+i])
						     	       /(dy[j_off+i]+dy[j_off+(i-1)]
						     	        +dy[j_1_off+i]+dy[j_1_off+(i-1)]) 
						     	      +(vb[k_off+j_off+i]
						     	     	-vb[k_off+j_off+(i-1)])
						     	       /(dx[j_off+i]+dx[j_off+(i-1)]
						     	        +dx[j_1_off+i]+dx[j_1_off+(i-1)]));

					xflux_advy_tmp = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]
											+dy[j_1_off+i]+dy[j_1_off+(i-1)])
						     		      *xflux_advy_tmp;
					xflux_advy[k_off+j_off+i] = xflux_advy_tmp;
				}

				if (i < imm1){

					yflux_advx_tmp = 0.125f*((dt[j_off+i]
						          			  +dt[j_1_off+i])
						          			*v[k_off+j_off+i] 
						          		   +(dt[j_off+(i-1)]
						          			  +dt[j_1_off+(i-1)])
						          			*v[k_off+j_off+(i-1)]) 
						          		 *(u[k_off+j_off+i]
						          		   +u[k_off+j_1_off+i]);

					yflux_advx_tmp -= dtaam
						     	     *((ub[k_off+j_off+i]
						     	          -ub[k_off+j_1_off+i])
						     	        /(dy[j_off+i]
						     	     	   +dy[j_off+(i-1)]
						     	     	   +dy[j_1_off+i]
						     	     	   +dy[j_1_off+(i-1)]) 
						     	      +(vb[k_off+j_off+i]
						     	     	 -vb[k_off+j_off+(i-1)])
						     	        /(dx[j_off+i]
						     	     	   +dx[j_off+(i-1)]
						     	     	   +dx[j_1_off+i]
						     	     	   +dx[j_1_off+(i-1)]));

					yflux_advx_tmp = 0.25f*(dx[j_off+i]
						     		     	+dx[j_off+(i-1)]
						     		     	+dx[j_1_off+i]
						     		     	+dx[j_1_off+(i-1)]) 
						     		      *yflux_advx_tmp;
					
					yflux_advx[k_off+j_off+i] = yflux_advx_tmp;
				}
			}
		}
	}
}

__global__ void
advct_ew_gpu_kernel_0(float * __restrict__ curv, 
				   float * __restrict__ xflux_advx,
				   float * __restrict__ yflux_advy,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ ub,
				   const float * __restrict__ vb,
				   const float * __restrict__ dt,
				   const float * __restrict__ aam,
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy,
				   int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	int kbm1 = kb-1;

	if (j < jm-1){
		for (k = 0; k < kbm1; k++){
			float xflux_advx_tmp, yflux_advy_tmp;

			curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]
										    +v[k_off+j_off+i])
										 *(dy[j_off+(i+1)]
											-dy[j_off+(i-1)]) 
										-(u[k_off+j_off+(i+1)]
											+u[k_off+j_off+i])
										 *(dx[j_A1_off+i]
											-dx[j_1_off+i])) 
										/(dx[j_off+i]*dy[j_off+i]);

			xflux_advx_tmp = 0.125f*((dt[j_off+(i+1)]
				          			   +dt[j_off+i])
				          			  *u[k_off+j_off+(i+1)] 
				          		    +(dt[j_off+i]
				          			   +dt[j_off+(i-1)])
				          			  *u[k_off+j_off+i]) 
				          		   *(u[k_off+j_off+(i+1)]
				          			 +u[k_off+j_off+i]);

			yflux_advy_tmp = 0.125f*((dt[j_A1_off+i]
				          			  +dt[j_off+i])
				          			 *v[k_off+j_A1_off+i] 
				          		   +(dt[j_off+i]
				          			  +dt[j_1_off+i])
				          			 *v[k_off+j_off+i]) 
				          		  *(v[k_off+j_A1_off+i] 
				          			 +v[k_off+j_off+i]);		

			xflux_advx_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				             *2.0f*(ub[k_off+j_off+(i+1)]
				     	     	 -ub[k_off+j_off+i])/dx[j_off+i];

			xflux_advx_tmp *= dy[j_off+i];

			yflux_advy_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				        *2.0f*(vb[k_off+j_A1_off+i]
				     		   -vb[k_off+j_off+i])
				        /dy[j_off+i];

			yflux_advy_tmp *= dx[j_off+i];

			xflux_advx[k_off+j_off+i] = xflux_advx_tmp;
			yflux_advy[k_off+j_off+i] = yflux_advy_tmp;

		}
	}
}

__global__ void
advct_sn_gpu_kernel_0(float * __restrict__ curv, 
				   float * __restrict__ xflux_advx,
				   float * __restrict__ yflux_advy,
				   const float * __restrict__ u, 
				   const float * __restrict__ v,
				   const float * __restrict__ ub,
				   const float * __restrict__ vb,
				   const float * __restrict__ dt,
				   const float * __restrict__ aam,
				   const float * __restrict__ dx, 
				   const float * __restrict__ dy,
				   int kb, int jm, int im){

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	int kbm1 = kb-1;

	if (i > 32 && i < im-33){ 
		for (k = 0; k < kbm1; k++){
			float xflux_advx_tmp, yflux_advy_tmp;

			curv[k_off+j_off+i] = 0.25f*((v[k_off+j_A1_off+i]
										    +v[k_off+j_off+i])
										 *(dy[j_off+(i+1)]
											-dy[j_off+(i-1)]) 
										-(u[k_off+j_off+(i+1)]
											+u[k_off+j_off+i])
										 *(dx[j_A1_off+i]
											-dx[j_1_off+i])) 
										/(dx[j_off+i]*dy[j_off+i]);

			xflux_advx_tmp = 0.125f*((dt[j_off+(i+1)]
				          			   +dt[j_off+i])
				          			  *u[k_off+j_off+(i+1)] 
				          		    +(dt[j_off+i]
				          			   +dt[j_off+(i-1)])
				          			  *u[k_off+j_off+i]) 
				          		   *(u[k_off+j_off+(i+1)]
				          			 +u[k_off+j_off+i]);

			yflux_advy_tmp = 0.125f*((dt[j_A1_off+i]
				          			  +dt[j_off+i])
				          			 *v[k_off+j_A1_off+i] 
				          		   +(dt[j_off+i]
				          			  +dt[j_1_off+i])
				          			 *v[k_off+j_off+i]) 
				          		  *(v[k_off+j_A1_off+i] 
				          			 +v[k_off+j_off+i]);		

			xflux_advx_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				             *2.0f*(ub[k_off+j_off+(i+1)]
				     	     	 -ub[k_off+j_off+i])/dx[j_off+i];

			xflux_advx_tmp *= dy[j_off+i];

			yflux_advy_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
				        *2.0f*(vb[k_off+j_A1_off+i]
				     		   -vb[k_off+j_off+i])
				        /dy[j_off+i];

			yflux_advy_tmp *= dx[j_off+i];

			xflux_advx[k_off+j_off+i] = xflux_advx_tmp;
			yflux_advy[k_off+j_off+i] = yflux_advy_tmp;

		}
	}
}

__global__ void
advct_ew_bcond_gpu_kernel_0(float * __restrict__ xflux_advx,
							int n_west,
							int kb, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int k;

	if (n_west == -1){
		if (j < jm-1){
			for (k = 0; k < kb-1; k++){
				xflux_advx[k_off+j_off] = 0;
			}
		}
	}


}

__global__ void
advct_sn_bcond_gpu_kernel_0(float* __restrict__ yflux_advy,
							int n_south,
							int kb, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int k;

	if (n_south == -1){
		if (i < im-1){
			for (k = 0; k < kb-1; k++){
				yflux_advy[k_off+i] = 0;
			}
		}
	}

}


__global__ void
advct_gpu_kernel_1(float * __restrict__ advx, 
				   float * __restrict__ advy, 
				   const float * __restrict__ xflux_advx,
				   const float * __restrict__ yflux_advx,
				   const float * __restrict__ xflux_advy,
				   const float * __restrict__ yflux_advy,
				   const float * __restrict__ curv,
				   const float * __restrict__ dt,
				   const float * __restrict__ u,
				   const float * __restrict__ v,
				   const float * __restrict__ aru,
				   const float * __restrict__ arv,
				   const int n_west, const int n_south, 
				   const int kb, const int jm, const int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y;
	const int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	if (n_west == -1){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advx_tmp;
				advx_tmp = xflux_advx[k_off+j_off+i]
					      -xflux_advx[k_off+j_off+(i-1)]
					      +yflux_advx[k_off+j_A1_off+i]
					      -yflux_advx[k_off+j_off+i];

				if (i > 1){
					advx_tmp = advx_tmp
						      -aru[j_off+i]*0.25f
						       *(curv[k_off+j_off+i]
						     	*dt[j_off+i]
						     	*(v[k_off+j_A1_off+i]
						     		+v[k_off+j_off+i]) 
						        +curv[k_off+j_off+i-1]
						         *dt[j_off+(i-1)]
						     	*(v[k_off+j_A1_off+(i-1)]
						     		+v[k_off+j_off+(i-1)]));	
				}

				advx[k_off+j_off+i] = advx_tmp;
			}
		}
	}else{
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advx_tmp;
				advx_tmp = xflux_advx[k_off+j_off+i]
					      -xflux_advx[k_off+j_off+(i-1)]
					      +yflux_advx[k_off+j_A1_off+i]
					      -yflux_advx[k_off+j_off+i];

				advx_tmp = advx_tmp
					      -aru[j_off+i]*0.25f
					     	*(curv[k_off+j_off+i]
					     	  *dt[j_off+i]
					     	  *(v[k_off+j_A1_off+i]
					     		  +v[k_off+j_off+i]) 
					     	 +curv[k_off+j_off+i-1]
					     	  *dt[j_off+(i-1)]
					     	  *(v[k_off+j_A1_off+(i-1)]
					     		  +v[k_off+j_off+(i-1)]));	

				advx[k_off+j_off+i] = advx_tmp;
			}
		}
	}

	if (n_south == -1){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advy_tmp;
				advy_tmp = xflux_advy[k_off+j_off+(i+1)]
					      -xflux_advy[k_off+j_off+i]
					      +yflux_advy[k_off+j_off+i]
					      -yflux_advy[k_off+j_1_off+i];

				if (j > 1){
					advy_tmp += arv[j_off+i]*0.25f
						       *(curv[k_off+j_off+i]
						     	  *dt[j_off+i]
						     	  *(u[k_off+j_off+(i+1)]
						     	   +u[k_off+j_off+i]) 
						        +curv[k_off+j_1_off+i]
						     	  *dt[j_1_off+i]
						     	  *(u[k_off+j_1_off+(i+1)]
						     	   +u[k_off+j_1_off+i]));	
				}

				advy[k_off+j_off+i] = advy_tmp;
			}
		}
	}else{
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advy_tmp;
				advy_tmp = xflux_advy[k_off+j_off+(i+1)]
					      -xflux_advy[k_off+j_off+i]
					      +yflux_advy[k_off+j_off+i]
					      -yflux_advy[k_off+j_1_off+i];

				advy_tmp += arv[j_off+i]*0.25f
					       *(curv[k_off+j_off+i]
					     	  *dt[j_off+i]
					     	  *(u[k_off+j_off+(i+1)]
					     		  +u[k_off+j_off+i]) 
					        +curv[k_off+j_1_off+i]
					     	  *dt[j_1_off+i]
					     	  *(u[k_off+j_1_off+(i+1)]
					     	   +u[k_off+j_1_off+i]));	
						
				advy[k_off+j_off+i] = advy_tmp;
			}
		}
	}

}


//__global__ void
//advct_gpu_kernel_1(float * __restrict__ xflux, 
//				   float * __restrict__ yflux,
//				   const float * __restrict__ u, 
//				   const float * __restrict__ v,
//				   const float * __restrict__ ub, 
//				   const float * __restrict__ vb, 
//				   const float * __restrict__ aam, 
//				   const float * __restrict__ dt, 
//				   const float * __restrict__ dx, 
//				   const float * __restrict__ dy, 
//				   int kb, int jm, int im){
//
//	//modify -xflux,yflux
//
//#ifdef D3_BLOCK
//	int k = blockDim.z*blockIdx.z + threadIdx.z;
//#else
//	int k;
//#endif
//
//	int j = blockDim.y*blockIdx.y + threadIdx.y;
//	int i = blockDim.x*blockIdx.x + threadIdx.x;
//
//	int kbm1 = kb-1;
//	//int jmm1 = jm-1;
//	int imm1 = im-1;
//
//	//float dtaam;
//
//#ifdef D3_BLOCK
//	float xflux_tmp, yflux_tmp;
//
//	if (k < kbm1 && j < jm & i > 0 && i < imm1){
//		xflux_tmp = 0.125f*((dt[j_off+(i+1)]+dt[j_off+i])*u[k_off+j_off+(i+1)] + 
//											(dt[j_off+i]+dt[j_off+(i-1)])*u[k_off+j_off+i]) *
//										(u[k_off+j_off+(i+1)]+u[k_off+j_off+i]);
//	}
//
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
//		yflux_tmp = 0.125f*((dt[j_off+i]+dt[j_1_off+i])*v[k_off+j_off+i] +
//								   (dt[j_off+(i-1)]+dt[j_1_off+(i-1)])*v[k_off+j_off+(i-1)]) *
//								  (u[k_off+j_off+i]+u[k_off+j_1_off+i]);
//	}
//
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < imm1){
//		dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]+dt[j_1_off+i]+dt[j_1_off+(i-1)])	* 
//							(aam[k_off+j_off+i]+aam[k_off+j_off+(i-1)] +
//								aam[k_off+j_1_off+i]+aam[k_off+j_1_off+(i-1)]);
//		xflux_tmp -= dt[j_off+i]*aam[k_off+j_off+i]*2.0f*
//									(ub[k_off+j_off+(i+1)]-ub[k_off+j_off+i])/dx[j_off+i];
//		xflux[k_off+j_off+i] = dy[j_off+i]*xflux_tmp;
//
//		yflux_tmp -= dtaam*((ub[k_off+j_off+i]-ub[k_off+j_1_off+i])/
//											(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)]) + 
//										(vb[k_off+j_off+i]-vb[k_off+j_off+(i-1)])/
//											(dx[j_off+i]+dx[j_off+(i-1)]+dx[j_1_off+i]+dx[j_1_off+(i-1)]));
//		yflux[k_off+j_off+i] = 0.25f*(dx[j_off+i]+dx[j_off+(i-1)]+dx[j_1_off+i]+dx[j_1_off+(i-1)]) *
//									yflux_tmp;
//	}
//
//#else
//	/*
//	for (k = 0; k < kbm1; k++){
//		for (j = 0; j < jm; j++){
//			for (i = 1; i < imm1; i++){
//				xflux[k][j][i] = 0.125f*((dt[j][i+1]+dt[j][i])*u[k][j][i+1] + (dt[j][i]+dt[j][i-1])*u[k][j][i])*(u[k][j][i+1]+u[k][j][i]);
//			}
//		}
//	}
//	*/
//
//	/*
//	if (k < kbm1 && j < jm && i > 0 && i < imm1)
//	{
//		xflux[k_off+j_off+i] = 0.125f*((dt[j_off+(i+1)]+dt[j_off+i])*u[k_off+j_off+(i+1)] + (dt[j_off+i]+dt[j_off+(i-1)])*u[k_off+j_off+i])*(u[k_off+j_off+(i+1)]+u[k_off+j_off+i]);
//
//	}
//	*/
//	
//	/*
//	if (j < jm && i > 0 && i < imm1){
//		for (k = 0; k < kbm1; k++){
//			xflux[k_off+j_off+i] = 0.125f*((dt[j_off+(i+1)]
//											  +dt[j_off+i])
//											*u[k_off+j_off+(i+1)] 
//										  +(dt[j_off+i]
//											  +dt[j_off+(i-1)])
//											*u[k_off+j_off+i]) 
//										 *(u[k_off+j_off+(i+1)]
//											+u[k_off+j_off+i]);
//		}
//	}
//	*/
//	
//
//	/*
//	for (k = 0; k < kbm1; k++){
//		for (j = 1; j < jm; j++){
//			for (i = 1; i < im; i++){
//				yflux[k][j][i] = 0.125f*((dt[j][i]+dt[j-1][i])*v[k][j][i] + (dt[j][i-1]+dt[j-1][i-1])*v[k][j][i-1])*(u[k][j][i]+u[k][j-1][i]);	
//			}
//		}
//	}
//	*/
//
//	
//	/*
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im)
//		yflux[k_off+j_off+i] = 0.125f*((dt[j_off+i]+dt[j_1_off+i])*v[k_off+j_off+i] +
//									   (dt[j_off+(i-1)]+dt[j_1_off+(i-1)])*v[k_off+j_off+(i-1)]) *
//									  (u[k_off+j_off+i]+u[k_off+j_1_off+i]);
//	*/
//
//	/*
//	if (j > 0 && j < jm && i > 0 && i < im){
//		for (k = 0; k < kbm1; k++){
//			yflux[k_off+j_off+i] = 0.125f*((dt[j_off+i]
//											  +dt[j_1_off+i])
//											*v[k_off+j_off+i] 
//										   +(dt[j_off+(i-1)]
//											  +dt[j_1_off+(i-1)])
//											*v[k_off+j_off+(i-1)]) 
//										 *(u[k_off+j_off+i]
//										   +u[k_off+j_1_off+i]);
//		}
//	}
//	*/
//
//	/*
//	for (k = 0; k < kbm1; k++){
//		for (j = 1; j < jm; j++){
//			for (i = 1; i < imm1; i++){
//				dtaam = 0.25f*(dt[j][i]+dt[j][i-1]+dt[j-1][i]+dt[j-1][i-1])*(aam[k][j][i]+aam[k][j][i-1]+aam[k][j-1][i]+aam[k][j-1][i-1]);
//
//				xflux[k][j][i] = xflux[k][j][i]-dt[j][i]*aam[k][j][i]*2.0f*(ub[k][j][i+1]-ub[k][j][i])/dx[j][i];
//				xflux[k][j][i] = dy[j][i]*xflux[k][j][i];
//
//				yflux[k][j][i] = yflux[k][j][i]-dtaam*((ub[k][j][i]-ub[k][j-1][i])/(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1])+(vb[k][j][i]-vb[k][j][i-1])/(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1]));
//				yflux[k][j][i] = 0.25f*(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1])*yflux[k][j][i];
//
//			}
//		}
//	}
//	*/
//
//	
//	/*
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < imm1){
//		dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]+dt[j_1_off+i]+dt[j_1_off+(i-1)])	* (aam[k_off+j_off+i]+aam[k_off+j_off+(i-1)]+aam[k_off+j_1_off+i]+aam[k_off+j_1_off+(i-1)]);
//		xflux[k_off+j_off+i] -= dt[j_off+i]*aam[k_off+j_off+i]*2.0f*(ub[k_off+j_off+(i+1)]-ub[k_off+j_off+i])/dx[j_off+i];
//		xflux[k_off+j_off+i] = dy[j_off+i]*xflux[k_off+j_off+i];
//
//		yflux[k_off+j_off+i] -= dtaam*((ub[k_off+j_off+i]-ub[k_off+j_1_off+i])/(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)]) + (vb[k_off+j_off+i]-vb[k_off+j_off+(i-1)])/(dx[j_off+i]+dx[j_off+(i-1)]+dx[j_1_off+i]+dx[j_1_off+(i-1)]));
//		yflux[k_off+j_off+i] = 0.25f*(dx[j_off+i]+dx[j_off+(i-1)]+dx[j_1_off+i]+dx[j_1_off+(i-1)])*yflux[k_off+j_off+i];
//	}
//	*/
//
//	/*
//	if (j > 0 && j < jm && i > 0 && i < imm1){
//		for (k = 0; k < kbm1; k++){
//			dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
//						    +dt[j_1_off+i]+dt[j_1_off+(i-1)])	
//						 *(aam[k_off+j_off+i]
//							+aam[k_off+j_off+(i-1)] 
//						    +aam[k_off+j_1_off+i]
//							+aam[k_off+j_1_off+(i-1)]);
//
//			xflux[k_off+j_off+i] -= dt[j_off+i]*aam[k_off+j_off+i]
//								   *2.0f*(ub[k_off+j_off+(i+1)]
//										 -ub[k_off+j_off+i])/dx[j_off+i];
//
//			xflux[k_off+j_off+i] = dy[j_off+i]*xflux[k_off+j_off+i];
//
//			yflux[k_off+j_off+i] -= dtaam
//									*((ub[k_off+j_off+i]
//									     -ub[k_off+j_1_off+i])
//									   /(dy[j_off+i]
//										   +dy[j_off+(i-1)]
//										   +dy[j_1_off+i]
//										   +dy[j_1_off+(i-1)]) 
//									 +(vb[k_off+j_off+i]
//										 -vb[k_off+j_off+(i-1)])
//									   /(dx[j_off+i]
//										   +dx[j_off+(i-1)]
//										   +dx[j_1_off+i]
//										   +dx[j_1_off+(i-1)]));
//
//			yflux[k_off+j_off+i] = 0.25f*(dx[j_off+i]
//											+dx[j_off+(i-1)]
//											+dx[j_1_off+i]
//											+dx[j_1_off+(i-1)]) 
//										*yflux[k_off+j_off+i];
//		}
//	}
//	*/
//
//	/*
//	if (j < jm && i > 0 && i < im){
//		for (k = 0; k < kbm1; k++){
//			float xflux_tmp, yflux_tmp;
//			float dtaam;
//			if (i < imm1){
//				xflux_tmp = 0.125f*((dt[j_off+(i+1)]
//					     			  +dt[j_off+i])
//					     			 *u[k_off+j_off+(i+1)] 
//					     		    +(dt[j_off+i]
//					     			  +dt[j_off+(i-1)])
//					     			 *u[k_off+j_off+i]) 
//					     		  *(u[k_off+j_off+(i+1)]
//					     			+u[k_off+j_off+i]);
//			}
//			
//			if (j > 0){
//				yflux_tmp = 0.125f*((dt[j_off+i]
//					     			  +dt[j_1_off+i])
//					     			*v[k_off+j_off+i] 
//					     		   +(dt[j_off+(i-1)]
//					     			  +dt[j_1_off+(i-1)])
//					     			*v[k_off+j_off+(i-1)]) 
//					     		 *(u[k_off+j_off+i]
//					     		   +u[k_off+j_1_off+i]);
//				
//			}
//
//			if (j > 0 && i < imm1){
//				dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
//							    +dt[j_1_off+i]+dt[j_1_off+(i-1)])	
//							 *(aam[k_off+j_off+i]
//								+aam[k_off+j_off+(i-1)] 
//							    +aam[k_off+j_1_off+i]
//								+aam[k_off+j_1_off+(i-1)]);
//
//				xflux_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
//					        *2.0f*(ub[k_off+j_off+(i+1)]
//					     		 -ub[k_off+j_off+i])/dx[j_off+i];
//
//				xflux_tmp *= dy[j_off+i];
//
//				yflux_tmp -= dtaam
//					     	*((ub[k_off+j_off+i]
//					     	     -ub[k_off+j_1_off+i])
//					     	   /(dy[j_off+i]
//					     		   +dy[j_off+(i-1)]
//					     		   +dy[j_1_off+i]
//					     		   +dy[j_1_off+(i-1)]) 
//					     	 +(vb[k_off+j_off+i]
//					     		 -vb[k_off+j_off+(i-1)])
//					     	   /(dx[j_off+i]
//					     		   +dx[j_off+(i-1)]
//					     		   +dx[j_1_off+i]
//					     		   +dx[j_1_off+(i-1)]));
//
//				yflux_tmp = 0.25f*(dx[j_off+i]
//					     			+dx[j_off+(i-1)]
//					     			+dx[j_1_off+i]
//					     			+dx[j_1_off+(i-1)]) 
//					     		*yflux_tmp;
//				
//			}
//
//			xflux[k_off+j_off+i] = xflux_tmp;
//			yflux[k_off+j_off+i] = yflux_tmp;
//		}
//	}
//	*/
//
//
//	#endif
//}

__global__ void
advct_gpu_kernel_2(float * __restrict__ advx, 
				   const float * __restrict__ curv, 
				   const float * __restrict__ xflux, 
				   const float * __restrict__ yflux,
				   const float * __restrict__ v, 
				   const float * __restrict__ dt,
				   const float * __restrict__ aru, 
				   int n_west,
				   int kb, int jm, int im){

	//modify -advx
#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	//float dtaam;

#ifdef D3_BLOCK
	float advx_tmp;

	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		advx_tmp = xflux[k_off+j_off+i]-xflux[k_off+j_off+(i-1)]+yflux[k_off+j_A1_off+i]-yflux[k_off+j_off+i];
	}

	if (n_west == -1){
		if (k < kbm1 && j > 0 && j < jmm1 && i > 1 && i < imm1){
			advx[k_off+j_off+i] = advx_tmp -aru[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(v[k_off+j_A1_off+i]+v[k_off+j_off+i]) + 
													   curv[k_off+j_off+i-1]*dt[j_off+(i-1)]*(v[k_off+j_A1_off+(i-1)]+v[k_off+j_off+(i-1)]));	
		}
	}else{
		if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
			advx[k_off+j_off+i] = advx_tmp -aru[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(v[k_off+j_A1_off+i]+v[k_off+j_off+i]) + 
													   curv[k_off+j_off+i-1]*dt[j_off+(i-1)]*(v[k_off+j_A1_off+(i-1)]+v[k_off+j_off+(i-1)]));	
		}
	}
#else
//! do horizontal advection
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				advx[k][j][i] = xflux[k][j][i]-xflux[k][j][i-1]+yflux[k][j+1][i]-yflux[k][j][i];	
			}
		}
	}
	*/
	
	/*
	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		advx[k_off+j_off+i] = xflux[k_off+j_off+i]-xflux[k_off+j_off+(i-1)]+yflux[k_off+j_A1_off+i]-yflux[k_off+j_off+i];
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 0; k < kbm1; k++){
			advx[k_off+j_off+i] = xflux[k_off+j_off+i]
								 -xflux[k_off+j_off+(i-1)]
								 +yflux[k_off+j_A1_off+i]
								 -yflux[k_off+j_off+i];
		}
	}
	*/
	

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			if (n_west == -1){
				for (i = 2; i < imm1; i++){
					advx[k][j][i] = advx[k][j][i]-aru[j][i]*0.25f*(curv[k][j][i]*dt[j][i]*(v[k][j+1][i]+v[k][j][i])+curv[k][j][i-1]*dt[j][i-1]*(v[k][j+1][i-1]+v[k][j][i-1]));
				}
			}else{
				for (i = 1; i < imm1; i++){
					advx[k][j][i] = advx[k][j][i]-aru[j][i]*0.25f*(curv[k][j][i]*dt[j][i]*(v[k][j+1][i]+v[k][j][i])+curv[k][j][i-1]*dt[j][i-1]*(v[k][j+1][i-1]+v[k][j][i-1]));
				}
			}
		}
	}
	*/

	/*
	if (k < kbm1 && j > 0 && j < jmm1){
		if (n_west == -1){
			if (i > 1 && i < imm1){
				advx[k_off+j_off+i] = advx[k_off+j_off+i] -aru[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(v[k_off+j_A1_off+i]+v[k_off+j_off+i]) + 
														   curv[k_off+j_off+i-1]*dt[j_off+(i-1)]*(v[k_off+j_A1_off+(i-1)]+v[k_off+j_off+(i-1)]));	
			}
		}else{
			if (i > 0 && i < imm1){
				advx[k_off+j_off+i] = advx[k_off+j_off+i] -aru[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(v[k_off+j_A1_off+i]+v[k_off+j_off+i]) + 
														   curv[k_off+j_off+i-1]*dt[j_off+(i-1)]*(v[k_off+j_A1_off+(i-1)]+v[k_off+j_off+(i-1)]));	
			}
		}
			
	}
	*/

	/*
	if (n_west == -1){
		if (j > 0 && j < jmm1 && i > 1 && i < imm1){
			for (k = 0; k < kbm1; k++){
				advx[k_off+j_off+i] = advx[k_off+j_off+i] 
									 -aru[j_off+i]*0.25f
									  *(curv[k_off+j_off+i]
										*dt[j_off+i]
										*(v[k_off+j_A1_off+i]
											+v[k_off+j_off+i]) 
									   +curv[k_off+j_off+i-1]
									    *dt[j_off+(i-1)]
										*(v[k_off+j_A1_off+(i-1)]
											+v[k_off+j_off+(i-1)]));	
			}
		}
	}else{
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				advx[k_off+j_off+i] = advx[k_off+j_off+i] 
									 -aru[j_off+i]*0.25f
										*(curv[k_off+j_off+i]
										  *dt[j_off+i]
										  *(v[k_off+j_A1_off+i]
											  +v[k_off+j_off+i]) 
										 +curv[k_off+j_off+i-1]
										  *dt[j_off+(i-1)]
										  *(v[k_off+j_A1_off+(i-1)]
											  +v[k_off+j_off+(i-1)]));	
			}
		}
	}
	*/

	if (n_west == -1){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advx_tmp;
				advx_tmp = xflux[k_off+j_off+i]
					      -xflux[k_off+j_off+(i-1)]
					      +yflux[k_off+j_A1_off+i]
					      -yflux[k_off+j_off+i];

				if (i > 1){
					advx_tmp = advx_tmp
						      -aru[j_off+i]*0.25f
						       *(curv[k_off+j_off+i]
						     	*dt[j_off+i]
						     	*(v[k_off+j_A1_off+i]
						     		+v[k_off+j_off+i]) 
						        +curv[k_off+j_off+i-1]
						         *dt[j_off+(i-1)]
						     	*(v[k_off+j_A1_off+(i-1)]
						     		+v[k_off+j_off+(i-1)]));	
				}

				advx[k_off+j_off+i] = advx_tmp;
			}
		}
	}else{
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advx_tmp;
				advx_tmp = xflux[k_off+j_off+i]
					      -xflux[k_off+j_off+(i-1)]
					      +yflux[k_off+j_A1_off+i]
					      -yflux[k_off+j_off+i];

				advx_tmp = advx_tmp
					      -aru[j_off+i]*0.25f
					     	*(curv[k_off+j_off+i]
					     	  *dt[j_off+i]
					     	  *(v[k_off+j_A1_off+i]
					     		  +v[k_off+j_off+i]) 
					     	 +curv[k_off+j_off+i-1]
					     	  *dt[j_off+(i-1)]
					     	  *(v[k_off+j_A1_off+(i-1)]
					     		  +v[k_off+j_off+(i-1)]));	

				advx[k_off+j_off+i] = advx_tmp;
			}
		}
	}

#endif
	
}

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
				   int kb, int jm, int im){

	//modify -advy, -xflux, -yflux
//#ifdef D3_BLOCK
//	int k = blockDim.z*blockIdx.z + threadIdx.z;
//#else
//	int k;
//#endif
//
//	int j = blockDim.y*blockIdx.y + threadIdx.y;
//	int i = blockDim.x*blockIdx.x + threadIdx.x;
//
//	int kbm1 = kb-1;
//	int jmm1 = jm-1;
//	//int imm1 = im-1;
//
//	//float dtaam;
//	
//#ifdef D3_BLOCK
//	float xflux_tmp, yflux_tmp;
//	/*
//	if (k < kb && j < jm && i < im){
//		advy[k_off+j_off+i] = 0;
//		xflux[k_off+j_off+i] = 0;
//		yflux[k_off+j_off+i] = 0;
//	}
//	*/
//
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
//		xflux_tmp = 0.125f*((dt[j_off+i]+dt[j_off+(i-1)])*u[k_off+j_off+i] + (dt[j_1_off+i]+dt[j_1_off+(i-1)])*u[k_off+j_1_off+i]) *
//								  (v[k_off+j_off+i]+v[k_off+j_off+(i-1)]);	
//	}
//
//	if (k < kbm1 && j > 0 && j < jmm1 && i < im){
//		yflux_tmp = 0.125f*((dt[j_A1_off+i]+dt[j_off+i])*v[k_off+j_A1_off+i] + (dt[j_off+i]+dt[j_1_off+i])*v[k_off+j_off+i]) *
//								  (v[k_off+j_A1_off+i] + v[k_off+j_off+i]);		
//	}
//
//	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < im){
//		dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]+dt[j_1_off+i]+dt[j_1_off+(i-1)]) *
//					  (aam[k_off+j_off+i]+aam[k_off+j_off+(i-1)]+aam[k_off+j_1_off+i]+aam[k_off+j_1_off+(i-1)]);
//		xflux_tmp -= dtaam*((ub[k_off+j_off+i]-ub[k_off+j_1_off+i])/(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)]) + 
//									   (vb[k_off+j_off+i]-vb[k_off+j_off+(i-1)])/(dx[j_off+i]+dx[j_off+(i-1)]+dx[j_1_off+i]+dx[j_1_off+(i-1)]));
//
//		xflux[k_off+j_off+i] = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)])*xflux_tmp;
//
//		yflux_tmp -= dt[j_off+i]*aam[k_off+j_off+i]*2.0f*(vb[k_off+j_A1_off+i]-vb[k_off+j_off+i])/dy[j_off+i];
//		yflux[k_off+j_off+i] = yflux_tmp*dx[j_off+i];
//	}
//
//#else
//! calculate y-component of velocity advection

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				advy[k][j][i] = 0;
				xflux[k][j][i] = 0;
				yflux[k][j][i] = 0;
			}
		}
	}
	*/
	
	/*
	if (k < kb && j < jm && i < im){
		advy[k_off+j_off+i] = 0.0f;
		xflux[k_off+j_off+i] = 0.0f;
		yflux[k_off+j_off+i] = 0.0f;
	}
	*/

	/*
	if (j < jm && i < im){
		for (k = 0; k < kb; k++){
			advy[k_off+j_off+i] = 0;
			xflux[k_off+j_off+i] = 0;
			yflux[k_off+j_off+i] = 0;
		}
	}
	*/
	

//! calculate horizontal advective fluxes
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i] = 0.125f*((dt[j][i]+dt[j][i-1])*u[k][j][i]+(dt[j-1][i]+dt[j-1][i-1])*u[k][j-1][i])*(v[k][j][i]+v[k][j][i-1]);
			}
		}
	}
	*/

	/*
	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
		xflux[k_off+j_off+i] = 0.125f*((dt[j_off+i]+dt[j_off+(i-1)])*u[k_off+j_off+i] + (dt[j_1_off+i]+dt[j_1_off+(i-1)])*u[k_off+j_1_off+i]) *
									  (v[k_off+j_off+i]+v[k_off+j_off+(i-1)]);	
	}
	*/

	/*
	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			xflux[k_off+j_off+i] = 0.125f*((dt[j_off+i]
											  +dt[j_off+(i-1)])
										    *u[k_off+j_off+i] 
										  +(dt[j_1_off+i]
											  +dt[j_1_off+(i-1)])
										    *u[k_off+j_1_off+i]) 
										 *(v[k_off+j_off+i]
											+v[k_off+j_off+(i-1)]);	
		}
	}
	*/
	

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 0; i < im; i++){
				yflux[k][j][i] = 0.125f*((dt[j+1][i]+dt[j][i])*v[k][j+1][i] + (dt[j][i]+dt[j-1][i])*v[k][j][i])*(v[k][j+1][i]+v[k][j][i]);
			}
		}
	}
	*/

	/*
	if (k < kbm1 && j > 0 && j < jmm1 && i < im){
		yflux[k_off+j_off+i] = 0.125f*((dt[j_A1_off+i]+dt[j_off+i])*v[k_off+j_A1_off+i] + (dt[j_off+i]+dt[j_1_off+i])*v[k_off+j_off+i]) *
									  (v[k_off+j_A1_off+i] + v[k_off+j_off+i]);		
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i < im){
		for (k = 0; k < kbm1; k++){
			yflux[k_off+j_off+i] = 0.125f*((dt[j_A1_off+i]
											  +dt[j_off+i])
											*v[k_off+j_A1_off+i] 
										  +(dt[j_off+i]
											  +dt[j_1_off+i])
											*v[k_off+j_off+i]) 
										 *(v[k_off+j_A1_off+i] 
											+v[k_off+j_off+i]);		
		}
	}
	*/

//! add horizontal diffusive fluxes
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < im; i++){
				dtaam = 0.25f*(dt[j][i]+dt[j][i-1]+dt[j-1][i]+dt[j-1][i-1])*(aam[k][j][i]+aam[k][j][i-1]+aam[k][j-1][i]+aam[k][j-1][i-1]);
				xflux[k][j][i] = xflux[k][j][i]-dtaam*((ub[k][j][i]-ub[k][j-1][i])/(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1]) + (vb[k][j][i]-vb[k][j][i-1])/(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1]));
				yflux[k][j][i] = yflux[k][j][i]-dt[j][i]*aam[k][j][i]*2.0f*(vb[k][j+1][i]-vb[k][j][i])/dy[j][i];
				xflux[k][j][i] = 0.25f*(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1])*xflux[k][j][i];
				yflux[k][j][i] = dx[j][i]*yflux[k][j][i];
			}
		}
	}
	*/

	/*
	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < im){
		dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]+dt[j_1_off+i]+dt[j_1_off+(i-1)]) *
					  (aam[k_off+j_off+i]+aam[k_off+j_off+(i-1)]+aam[k_off+j_1_off+i]+aam[k_off+j_1_off+(i-1)]);
		xflux[k_off+j_off+i] -= dtaam*((ub[k_off+j_off+i]-ub[k_off+j_1_off+i])/(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)]) + 
									   (vb[k_off+j_off+i]-vb[k_off+j_off+(i-1)])/(dx[j_off+i]+dx[j_off+(i-1)]+dx[j_1_off+i]+dx[j_1_off+(i-1)]));

		xflux[k_off+j_off+i] = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)])*xflux[k_off+j_off+i];
		//xflux[k_off+j_off+i] *= 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]+dy[j_1_off+i]+dy[j_1_off+(i-1)]);

		yflux[k_off+j_off+i] -= dt[j_off+i]*aam[k_off+j_off+i]*2.0f*(vb[k_off+j_A1_off+i]-vb[k_off+j_off+i])/dy[j_off+i];
		yflux[k_off+j_off+i] *= dx[j_off+i];
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
						  +dt[j_1_off+i]+dt[j_1_off+(i-1)]) 
						 *(aam[k_off+j_off+i]
							+aam[k_off+j_off+(i-1)]
							+aam[k_off+j_1_off+i]
							+aam[k_off+j_1_off+(i-1)]);

			xflux[k_off+j_off+i] -= dtaam
									*((ub[k_off+j_off+i]
									    -ub[k_off+j_1_off+i])
									  /(dy[j_off+i]+dy[j_off+(i-1)]
									   +dy[j_1_off+i]+dy[j_1_off+(i-1)]) 
									 +(vb[k_off+j_off+i]
										-vb[k_off+j_off+(i-1)])
									  /(dx[j_off+i]+dx[j_off+(i-1)]
									   +dx[j_1_off+i]+dx[j_1_off+(i-1)]));

			xflux[k_off+j_off+i] = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]
										 +dy[j_1_off+i]+dy[j_1_off+(i-1)])
										*xflux[k_off+j_off+i];

			yflux[k_off+j_off+i] -= dt[j_off+i]*aam[k_off+j_off+i]
								   *2.0f*(vb[k_off+j_A1_off+i]
										   -vb[k_off+j_off+i])
								   /dy[j_off+i];

			yflux[k_off+j_off+i] *= dx[j_off+i];
		}
	}
	*/

	/*
	if (j > 0 && j < jm && i < im){
		for (k = 0; k < kbm1; k++){
			float xflux_tmp, yflux_tmp;
			float dtaam;
			if (i > 0){
				xflux_tmp = 0.125f*((dt[j_off+i]
					      			  +dt[j_off+(i-1)])
					      		    *u[k_off+j_off+i] 
					      		  +(dt[j_1_off+i]
					      			  +dt[j_1_off+(i-1)])
					      		    *u[k_off+j_1_off+i]) 
					      		 *(v[k_off+j_off+i]
					      			+v[k_off+j_off+(i-1)]);	
			}
			if (j < jmm1){
				yflux_tmp = 0.125f*((dt[j_A1_off+i]
					     			  +dt[j_off+i])
					     			 *v[k_off+j_A1_off+i] 
					     		   +(dt[j_off+i]
					     			  +dt[j_1_off+i])
					     			 *v[k_off+j_off+i]) 
					     		  *(v[k_off+j_A1_off+i] 
					     			 +v[k_off+j_off+i]);		
			}

			if (j < jmm1 && i > 0){
				dtaam = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)]
							  +dt[j_1_off+i]+dt[j_1_off+(i-1)]) 
							 *(aam[k_off+j_off+i]
								+aam[k_off+j_off+(i-1)]
								+aam[k_off+j_1_off+i]
								+aam[k_off+j_1_off+(i-1)]);

				xflux_tmp -= dtaam
					     	*((ub[k_off+j_off+i]
					     	    -ub[k_off+j_1_off+i])
					     	  /(dy[j_off+i]+dy[j_off+(i-1)]
					     	   +dy[j_1_off+i]+dy[j_1_off+(i-1)]) 
					     	 +(vb[k_off+j_off+i]
					     		-vb[k_off+j_off+(i-1)])
					     	  /(dx[j_off+i]+dx[j_off+(i-1)]
					     	   +dx[j_1_off+i]+dx[j_1_off+(i-1)]));

				xflux_tmp = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)]
					     		 +dy[j_1_off+i]+dy[j_1_off+(i-1)])
					     		*xflux_tmp;

				yflux_tmp -= dt[j_off+i]*aam[k_off+j_off+i]
					        *2.0f*(vb[k_off+j_A1_off+i]
					     		   -vb[k_off+j_off+i])
					        /dy[j_off+i];

				yflux_tmp *= dx[j_off+i];
			}

			xflux[k_off+j_off+i] = xflux_tmp;
			yflux[k_off+j_off+i] = yflux_tmp;
		}
	}
	*/
//#endif
}

__global__ void
advct_gpu_kernel_4(float * __restrict__ advy, 
				   const float * __restrict__ curv, 
				   const float * __restrict__ xflux, 
				   const float * __restrict__ yflux,
				   const float * __restrict__ u, 
				   const float * __restrict__ dt, 
				   const float * __restrict__ arv, 
				   int n_south, 
				   int kb, int jm, int im){

	//modify -advy
#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;


#ifdef D3_BLOCK
	float advy_tmp;

	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		advy_tmp = xflux[k_off+j_off+(i+1)]-xflux[k_off+j_off+i]+yflux[k_off+j_off+i]-yflux[k_off+j_1_off+i];
	}

	if (n_south == -1){
		if (k < kbm1 && j > 1 && j < jmm1 && i > 0 && i < imm1){
			advy[k_off+j_off+i] = advy_tmp + arv[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(u[k_off+j_off+(i+1)]+u[k_off+j_off+i]) +
													   curv[k_off+j_1_off+i]*dt[j_1_off+i]*(u[k_off+j_1_off+(i+1)]+u[k_off+j_1_off+i]));	
					
		}
	}else{
		if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
			advy[k_off+j_off+i] = advy_tmp + arv[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(u[k_off+j_off+(i+1)]+u[k_off+j_off+i]) +
													   curv[k_off+j_1_off+i]*dt[j_1_off+i]*(u[k_off+j_1_off+(i+1)]+u[k_off+j_1_off+i]));	
					
		}
	}
#else
//! do horizontal advection
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				advy[k][j][i] = xflux[k][j][i+1]-xflux[k][j][i]+yflux[k][j][i]-yflux[k][j-1][i];	
			}
		}
	}
	*/

	/*
	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		advy[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]-xflux[k_off+j_off+i]+yflux[k_off+j_off+i]-yflux[k_off+j_1_off+i];
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 0; k < kbm1; k++){
			advy[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]
								 -xflux[k_off+j_off+i]
								 +yflux[k_off+j_off+i]
								 -yflux[k_off+j_1_off+i];
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		for (i = 1; i < imm1; i++){
			if (n_south == -1){
				for (j = 2; j < jmm1; j++){
					advy[k][j][i] = advy[k][j][i]+arv[j][i]*0.25f*(curv[k][j][i]*dt[j][i]*(u[k][j][i+1]+u[k][j][i]) + curv[k][j-1][i]*dt[j-1][i]*(u[k][j-1][i+1]+u[k][j-1][i]));
				}
			}else{
				for (j = 1; j < jmm1; j++){
					advy[k][j][i] = advy[k][j][i]+arv[j][i]*0.25f*(curv[k][j][i]*dt[j][i]*(u[k][j][i+1]+u[k][j][i]) + curv[k][j-1][i]*dt[j-1][i]*(u[k][j-1][i+1]+u[k][j-1][i]));	
				}
			}
		}
	}
	*/

	/*
	if (k < kbm1 && i > 0 && i < imm1){
		if (n_south == -1){
			if (j > 1 && j < jmm1){
				advy[k_off+j_off+i] += arv[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(u[k_off+j_off+(i+1)]+u[k_off+j_off+i]) +
														   curv[k_off+j_1_off+i]*dt[j_1_off+i]*(u[k_off+j_1_off+(i+1)]+u[k_off+j_1_off+i]));	
			}
		}else{
			if (j > 0 && j < jmm1){
				advy[k_off+j_off+i] += arv[j_off+i]*0.25f*(curv[k_off+j_off+i]*dt[j_off+i]*(u[k_off+j_off+(i+1)]+u[k_off+j_off+i]) +
														   curv[k_off+j_1_off+i]*dt[j_1_off+i]*(u[k_off+j_1_off+(i+1)]+u[k_off+j_1_off+i]));	
			}
		}
	}
	*/

	if (n_south == -1){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advy_tmp;
				advy_tmp = xflux[k_off+j_off+(i+1)]
					      -xflux[k_off+j_off+i]
					      +yflux[k_off+j_off+i]
					      -yflux[k_off+j_1_off+i];

				if (j > 1){
					advy_tmp += arv[j_off+i]*0.25f
						       *(curv[k_off+j_off+i]
						     	  *dt[j_off+i]
						     	  *(u[k_off+j_off+(i+1)]
						     	   +u[k_off+j_off+i]) 
						        +curv[k_off+j_1_off+i]
						     	  *dt[j_1_off+i]
						     	  *(u[k_off+j_1_off+(i+1)]
						     	   +u[k_off+j_1_off+i]));	
				}

				advy[k_off+j_off+i] = advy_tmp;
			}
		}
	}else{
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			for (k = 0; k < kbm1; k++){
				float advy_tmp;
				advy_tmp = xflux[k_off+j_off+(i+1)]
					      -xflux[k_off+j_off+i]
					      +yflux[k_off+j_off+i]
					      -yflux[k_off+j_1_off+i];

				advy_tmp += arv[j_off+i]*0.25f
					       *(curv[k_off+j_off+i]
					     	  *dt[j_off+i]
					     	  *(u[k_off+j_off+(i+1)]
					     		  +u[k_off+j_off+i]) 
					        +curv[k_off+j_1_off+i]
					     	  *dt[j_1_off+i]
					     	  *(u[k_off+j_1_off+(i+1)]
					     	   +u[k_off+j_1_off+i]));	
						
				advy[k_off+j_off+i] = advy_tmp;
			}
		}
	}

#endif

}

/*
void advct_gpu(float advx[][j_size][i_size], float v[][j_size][i_size],
			   float u[][j_size][i_size], float dt[][i_size], 
			   float ub[][j_size][i_size], float aam[][j_size][i_size],
			   float vb[][j_size][i_size], float advy[][j_size][i_size]){
*/

/*
void advct_gpu(float *d_advx,
			   float *d_advy,
		       float *d_u, float *d_ub, 
			   float *d_v, float *d_vb, 
			   float *d_dt, float *d_aam){
*/
void advct_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_advct,
				   end_advct;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advct);
#endif

/*modify: -advx/advy
 *
 *
 */

	float *d_xflux_advx = d_3d_tmp0;
	float *d_xflux_advx_east = d_3d_tmp0_east;
	float *d_xflux_advx_west = d_3d_tmp0_west;
	float *d_xflux_advx_south = d_3d_tmp0_south;
	float *d_xflux_advx_north = d_3d_tmp0_north;

	float *d_yflux_advy = d_3d_tmp1;
	float *d_yflux_advy_east = d_3d_tmp1_east;
	float *d_yflux_advy_west = d_3d_tmp1_west;
	float *d_yflux_advy_south = d_3d_tmp1_south;
	float *d_yflux_advy_north = d_3d_tmp1_north;

	float *d_curv = d_3d_tmp2;
	float *d_curv_east = d_3d_tmp2_east;
	float *d_curv_west = d_3d_tmp2_west;
	float *d_curv_south = d_3d_tmp2_south;
	float *d_curv_north = d_3d_tmp2_north;

	float *d_xflux_advy = d_3d_tmp3;
	float *d_yflux_advx = d_3d_tmp4;
	
    //float xflux[k_size][j_size][i_size];
	//float yflux[k_size][j_size][i_size];
    //float curv[k_size][j_size][i_size];
    //float dtaam;
	

#ifdef D3_BLOCK
	dim3 threadPerBlock(block_i_3D, block_j_3D, block_k_3D);
	dim3 blockPerGrid((im+block_i_3D-1)/block_i_3D, (jm+block_j_3D-1)/block_j_3D, (kb+block_k_3D-1)/block_k_3D);
	dim3 threadPerBlock_3x2(block_i_3D, block_k_3D, block_j_3D);
	dim3 blockPerGrid_3x2((im+block_i_3D-1)/block_i_3D, (jm+block_k_3D-1)/block_k_3D, (kb+block_j_3D-1)/block_j_3D);
#else
	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);
#endif

	dim3 threadPerBlock_ew(32, 4);
	dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	dim3 threadPerBlock_ew_bcond(1, 128);
	dim3 blockPerGrid_ew_bcond(1, (j_size-2+127)/128);

	dim3 threadPerBlock_sn(32, 4);
	dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	dim3 threadPerBlock_sn_bcond(128, 1);
	dim3 blockPerGrid_sn_bcond((i_size-2+127)/128, 1);

	
	/*
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ub, ub, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vb, vb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/

	
	//modify -curv
	//advct_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		d_curv, d_xflux_advx, d_xflux_advy, d_yflux_advx, d_yflux_advy,
	//		d_u, d_v, d_ub, d_vb, d_dt, d_aam,
	//		d_dx, d_dy, kb, jm, im);

/*
#ifdef CUDA_SLICE_MPI
    exchange3d_mpi_gpu(d_curv,im,jm,kbm1);
#else
    float curv[k_size][j_size][i_size];
	checkCudaErrors(cudaMemcpy(curv, d_curv, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));

    //exchange3d_mpi(curv(:,:,1:kbm1),im,jm,kbm1)
    //exchange3d_mpi_xsz_(curv,im,jm,0,kbm1-1);
    exchange3d_mpi(curv,im,jm,kbm1);

	checkCudaErrors(cudaMemcpy(d_curv, curv, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
#endif
*/
    //exchange3d_mpi_gpu(d_curv,im,jm,kbm1);
    //exchange3d_mpi_gpu(d_xflux_advx,im,jm,kbm1);
    //exchange3d_mpi_gpu(d_yflux_advy,im,jm,kbm1);

    //exchange3d_cuda_aware_mpi(d_curv,im,jm,kbm1);
    //exchange3d_cuda_aware_mpi(d_xflux_advx,im,jm,kbm1);
    //exchange3d_cuda_aware_mpi(d_yflux_advy,im,jm,kbm1);

	advct_ew_gpu_kernel_0<<<blockPerGrid_ew, threadPerBlock_ew,
							   0, stream[1]>>>(
			d_curv, d_xflux_advx, d_yflux_advy,
			d_u, d_v, d_ub, d_vb, d_dt, d_aam,
			d_dx, d_dy, kb, jm, im);

	advct_sn_gpu_kernel_0<<<blockPerGrid_sn, threadPerBlock_sn,
							   0, stream[2]>>>(
			d_curv, d_xflux_advx, d_yflux_advy,
			d_u, d_v, d_ub, d_vb, d_dt, d_aam,
			d_dx, d_dy, kb, jm, im);

	advct_inner_gpu_kernel_0<<<blockPerGrid, threadPerBlock,
							   0, stream[0]>>>(
			d_curv, d_xflux_advx, d_xflux_advy, d_yflux_advx, d_yflux_advy,
			d_u, d_v, d_ub, d_vb, d_dt, d_aam,
			d_dx, d_dy, kb, jm, im);

	advct_ew_bcond_gpu_kernel_0<<<blockPerGrid_ew_bcond, 
								  threadPerBlock_ew_bcond,
								  0, stream[3]>>>(
			d_xflux_advx, n_west, kb, jm, im);

	advct_sn_bcond_gpu_kernel_0<<<blockPerGrid_sn_bcond, 
								  threadPerBlock_sn_bcond,
							      0, stream[4]>>>(
			d_yflux_advy, n_south, kb, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));


    exchange3d_cudaUVA(d_curv, 
					   d_curv_east, d_curv_west, 
					   d_curv_south, d_curv_north, 
					   stream[1], im,jm,kbm1);

    exchange3d_cudaUVA(d_xflux_advx, 
					   d_xflux_advx_east, d_xflux_advx_west, 
					   d_xflux_advx_south, d_xflux_advx_north, 
					   stream[1], im,jm,kbm1);

    exchange3d_cudaUVA(d_yflux_advy, 
					   d_yflux_advy_east, d_yflux_advy_west, 
					   d_yflux_advy_south, d_yflux_advy_north,
					   stream[1], im,jm,kbm1);

	//MPI_Barrier(pom_comm);
    //exchange3d_cuda_ipc(d_curv, d_curv_east, d_curv_west,
	//					stream[1], im,jm,kbm1);
    //exchange3d_cuda_ipc(d_xflux_advx, d_xflux_advx_east, d_xflux_advx_west,
	//					stream[1], im,jm,kbm1);
    //exchange3d_cuda_ipc(d_yflux_advy, d_yflux_advy_east, d_yflux_advy_west,
	//					stream[1], im,jm,kbm1);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[3]));
	checkCudaErrors(cudaStreamSynchronize(stream[4]));
	checkCudaErrors(cudaStreamSynchronize(stream[0]));


	advct_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_advx, d_advy,
			d_xflux_advx, d_yflux_advx, 
			d_xflux_advy, d_yflux_advy, 
			d_curv, d_dt, d_u, d_v, 
			d_aru, d_arv,
			n_west, n_south, kb, jm, im);


	
	/*
	checkCudaErrors(cudaMemcpy(advx, d_advx, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(advy, d_advy, kb*jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));
	*/
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advct);
		advct_time += time_consumed(&start_advct, 
								    &end_advct);
#endif
	
	return;
}




__global__ void
baropg_gpu_kernel_0(float * __restrict__ rho, 
					const float * __restrict__ rmean,
					int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	//int kbm1 = kb-1;
	//int jmm1 = jm-1;
	//int imm1 = im-1;

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rho[k][j][i]-=rmean[k][j][i];	
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			rho[k_off+j_off+i] -= rmean[k_off+j_off+i];	
		}
	}
}

#ifdef INIT_VERSION
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
					int kb, int jm, int im){
	//modify -drhox, -drhoy(after test, the boundary of drhox is of no use)
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			drhox[0][j][i]=0.5f*grav*(-zz[0])*(dt[j][i]+dt[j][i-1])
							   *(rho[0][j][i]-rho[0][j][i-1]);
		}
	}
	*/

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		drhox[j_off+i] = 0.5f*grav*(-zz[0])*(dt[j_off+i]+dt[j_off+(i-1)])
							 *(rho[j_off+i]-rho[j_off+(i-1)]);
	}

	/*
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i]=drhox[k-1][j][i]
								+grav*0.25f*(zz[k-1]-zz[k])
									*(dt[j][i]+dt[j][i-1])
									*(rho[k][j][i]-rho[k][j][i-1]
										+rho[k-1][j][i]-rho[k-1][j][i-1])
								+grav*0.25f*(zz[k-1]+zz[k])
									*(dt[j][i]-dt[j][i-1])
									*(rho[k][j][i]+rho[k][j][i-1]
										-rho[k-1][j][i]-rho[k-1][j][i-1]);
			}
		}
	}
	*/

	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhox[k_off+j_off+i] = drhox[k_1_off+j_off+i] 
								  +grav*0.25f*(zz[k-1]-zz[k])
									   *(dt[j_off+i]+dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										-rho[k_off+j_off+(i-1)]
										+rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]) 
								  +grav*0.25f*(zz[k-1]+zz[k])
									   *(dt[j_off+i]-dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										+rho[k_off+j_off+(i-1)]
										-rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]);
		}
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i]=0.25f*(dt[j][i]+dt[j][i-1])
								   *drhox[k][j][i]*dum[j][i]
								   *(dy[j][i]+dy[j][i-1]);
			}
		}
	}
	*/

	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhox[k_off+j_off+i] = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)])
										*drhox[k_off+j_off+i]
										*dum[j_off+i]
										*(dy[j_off+i]+dy[j_off+(i-1)]);
		}
	}

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			drhoy[0][j][i]=0.5f*grav*(-zz[0])*(dt[j][i]+dt[j-1][i])
							   *(rho[0][j][i]-rho[0][j-1][i]);
		}
	}
	*/
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		drhoy[j_off+i] = 0.5f*grav*(-zz[0])
						     *(dt[j_off+i]+dt[j_1_off+i])
							 *(rho[j_off+i]-rho[j_1_off+i]);
	}

	/*
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1;i < imm1; i++){
				drhoy[k][j][i]=drhoy[k-1][j][i]	
								+grav*0.25f*(zz[k-1]-zz[k])
									 *(dt[j][i]+dt[j-1][i])
									 *(rho[k][j][i]-rho[k][j-1][i]
										+rho[k-1][j][i]-rho[k-1][j-1][i])
								+grav*0.25f*(zz[k-1]+zz[k])
									 *(dt[j][i]-dt[j-1][i])
									 *(rho[k][j][i]+rho[k][j-1][i]
										-rho[k-1][j][i]-rho[k-1][j-1][i]);
			}
		}
	}
	*/
	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhoy[k_off+j_off+i] = drhoy[k_1_off+j_off+i] 
									+grav*0.25f*(zz[k-1]-zz[k])
										 *(dt[j_off+i]+dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  -rho[k_off+j_1_off+i] 
										  +rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]) 
									+grav*0.25f*(zz[k-1]+zz[k])
										 *(dt[j_off+i]-dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  +rho[k_off+j_1_off+i]
										  -rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]);
		}
	}

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhoy[k][j][i]=0.25f*(dt[j][i]+dt[j-1][i])
									*drhoy[k][j][i]*dvm[j][i]
									*(dx[j][i]+dx[j-1][i]);
			}
		}
	}
	*/
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhoy[k_off+j_off+i] = 0.25f*(dt[j_off+i]+dt[j_1_off+i])
										*drhoy[k_off+j_off+i]
										*dvm[j_off+i]
										*(dx[j_off+i]+dx[j_1_off+i]);
		}
	}

	/*
	for (k = 0; k < kb; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i] *= (ramp);	
				drhoy[k][j][i] *= (ramp);	
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhox[k_off+j_off+i] *= ramp;
			drhoy[k_off+j_off+i] *= ramp;
		}
	}
}

#endif

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
					int kb, int jm, int im){

	//modify -drhox(after test, the boundary of drhox is of no use)
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	//float drhox_tmp[k_size], drhoy_tmp[k_size];

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		drhox[j_off+i] = 0.5f*grav*(-zz[0])
							 *(dt[j_off+i]+dt[j_off+(i-1)])
							 *(rho[j_off+i]-rho[j_off+(i-1)]);

		drhoy[j_off+i] = 0.5f*grav*(-zz[0])
						     *(dt[j_off+i]+dt[j_1_off+i])
							 *(rho[j_off+i]-rho[j_1_off+i]);
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		drhox_tmp[0] = 0.5f*grav*(-zz[0])
							 *(dt[j_off+i]+dt[j_off+(i-1)])
							 *(rho[j_off+i]-rho[j_off+(i-1)]);

		drhoy_tmp[0] = 0.5f*grav*(-zz[0])
						     *(dt[j_off+i]+dt[j_1_off+i])
							 *(rho[j_off+i]-rho[j_1_off+i]);
	}
	*/

	/*
	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhox[k_off+j_off+i] = drhox[k_1_off+j_off+i] 
								  +grav*0.25f*(zz[k-1]-zz[k])
									   *(dt[j_off+i]+dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										-rho[k_off+j_off+(i-1)]
										+rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]) 
								  +grav*0.25f*(zz[k-1]+zz[k])
									   *(dt[j_off+i]-dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										+rho[k_off+j_off+(i-1)]
										-rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]);

			drhoy[k_off+j_off+i] = drhoy[k_1_off+j_off+i] 
									+grav*0.25f*(zz[k-1]-zz[k])
										 *(dt[j_off+i]+dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  -rho[k_off+j_1_off+i] 
										  +rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]) 
									+grav*0.25f*(zz[k-1]+zz[k])
										 *(dt[j_off+i]-dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  +rho[k_off+j_1_off+i]
										  -rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]);
		}
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 1; k < kbm1; k++){
			drhox_tmp[k] = drhox_tmp[k-1] 
								  +grav*0.25f*(zz[k-1]-zz[k])
									   *(dt[j_off+i]+dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										-rho[k_off+j_off+(i-1)]
										+rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]) 
								  +grav*0.25f*(zz[k-1]+zz[k])
									   *(dt[j_off+i]-dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										+rho[k_off+j_off+(i-1)]
										-rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]);

			drhoy_tmp[k] = drhoy_tmp[k-1] 
									+grav*0.25f*(zz[k-1]-zz[k])
										 *(dt[j_off+i]+dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  -rho[k_off+j_1_off+i] 
										  +rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]) 
									+grav*0.25f*(zz[k-1]+zz[k])
										 *(dt[j_off+i]-dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  +rho[k_off+j_1_off+i]
										  -rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]);
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			drhox[k_off+j_off+i] = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)])
										*drhox[k_off+j_off+i]
										*dum[j_off+i]
										*(dy[j_off+i]+dy[j_off+(i-1)]);

			drhoy[k_off+j_off+i] = 0.25f*(dt[j_off+i]+dt[j_1_off+i])
										*drhoy[k_off+j_off+i]
										*dvm[j_off+i]
										*(dx[j_off+i]+dx[j_1_off+i]);

			drhox[k_off+j_off+i] *= ramp;
			drhoy[k_off+j_off+i] *= ramp;
		}
	}
	*/

	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 0; k < kbm1; k++){
			drhox_tmp[k] = 0.25f*(dt[j_off+i]+dt[j_off+(i-1)])
										*drhox_tmp[k]
										*dum[j_off+i]
										*(dy[j_off+i]+dy[j_off+(i-1)]);

			drhoy_tmp[k] = 0.25f*(dt[j_off+i]+dt[j_1_off+i])
										*drhoy_tmp[k]
										*dvm[j_off+i]
										*(dx[j_off+i]+dx[j_1_off+i]);

			drhox[k_off+j_off+i] = drhox_tmp[k]*ramp;
			drhoy[k_off+j_off+i] = drhoy_tmp[k]*ramp;
		}
	}
	*/

	////////////////////////////////////////////
	////////////////////////////////////////////

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		drhox[j_off+i] = 0.5f*grav*(-zz[0])
							 *(dt[j_off+i]+dt[j_off+(i-1)])
							 *(rho[j_off+i]-rho[j_off+(i-1)]);

		drhoy[j_off+i] = 0.5f*grav*(-zz[0])
						     *(dt[j_off+i]+dt[j_1_off+i])
							 *(rho[j_off+i]-rho[j_1_off+i]);

		for (k = 1; k < kbm1; k++){
			drhox[k_off+j_off+i] = drhox[k_1_off+j_off+i] 
								  +grav*0.25f*(zz[k-1]-zz[k])
									   *(dt[j_off+i]+dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										-rho[k_off+j_off+(i-1)]
										+rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]) 
								  +grav*0.25f*(zz[k-1]+zz[k])
									   *(dt[j_off+i]-dt[j_off+(i-1)])
									   *(rho[k_off+j_off+i]
										+rho[k_off+j_off+(i-1)]
										-rho[k_1_off+j_off+i]
										-rho[k_1_off+j_off+(i-1)]);

			drhoy[k_off+j_off+i] = drhoy[k_1_off+j_off+i] 
									+grav*0.25f*(zz[k-1]-zz[k])
										 *(dt[j_off+i]+dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  -rho[k_off+j_1_off+i] 
										  +rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]) 
									+grav*0.25f*(zz[k-1]+zz[k])
										 *(dt[j_off+i]-dt[j_1_off+i])
										 *(rho[k_off+j_off+i]
										  +rho[k_off+j_1_off+i]
										  -rho[k_1_off+j_off+i]
										  -rho[k_1_off+j_1_off+i]);

			drhox[k_1_off+j_off+i] = (0.25f*(dt[j_off+i]+dt[j_off+(i-1)])
										*drhox[k_1_off+j_off+i]
										*dum[j_off+i]
										*(dy[j_off+i]+dy[j_off+(i-1)]))
									 *ramp;

			drhoy[k_1_off+j_off+i] = (0.25f*(dt[j_off+i]+dt[j_1_off+i])
										*drhoy[k_1_off+j_off+i]
										*dvm[j_off+i]
										*(dx[j_off+i]+dx[j_1_off+i]))
									 *ramp;
		}

		drhox[k_1_off+j_off+i] = (0.25f*(dt[j_off+i]+dt[j_off+(i-1)])
									*drhox[k_1_off+j_off+i]
									*dum[j_off+i]
									*(dy[j_off+i]+dy[j_off+(i-1)]))
								 *ramp;

		drhoy[k_1_off+j_off+i] = (0.25f*(dt[j_off+i]+dt[j_1_off+i])
									*drhoy[k_1_off+j_off+i]
									*dvm[j_off+i]
									*(dx[j_off+i]+dx[j_1_off+i]))
								 *ramp;
	}
}

__global__ void
baropg_gpu_kernel_2(float * __restrict__ rho, 
					const float * __restrict__ rmean,
					int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	//int kbm1 = kb-1;
	//int jmm1 = jm-1;
	//int imm1 = im-1;

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rho[k][j][i] += rmean[k][j][i];	
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			rho[k_off+j_off+i] += rmean[k_off+j_off+i];	
		}
	}
}

/*
void baropg_gpu(float rho[][j_size][i_size], float drhox[][j_size][i_size],
				float dt[][i_size], float drhoy[][j_size][i_size], 
				float ramp){
*/

/*
void baropg_gpu(float *d_drhox, float *d_drhoy,
				float *d_dt, float *d_rho, 
				float ramp){
*/
void baropg_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_baropg,
				   end_baropg;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_baropg);
#endif

/*change: -drhox,-drhoy (after testing, the values on the boundary of drhox/drhoy
		                 are of no use)
 *
 */

	//float ramp = *ramp_c;
	//int i,j,k;

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
			          (j_size+block_j_2D-1)/block_j_2D);
	
	/*
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rho, rho, kb*jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	*/
	

	baropg_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_rho, d_rmean, 
			kb, jm, im);

//! calculate x-component of baroclinic pressure gradient
//! calculate y-component of baroclinic pressure gradient

	//checkCudaErrors(cudaDeviceSynchronize());

	baropg_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_drhox, d_drhoy, d_dt, d_rho, 
		    d_dx, d_dy, d_dum, d_dvm, d_zz,
		    grav, ramp, kb, jm, im);

	//checkCudaErrors(cudaDeviceSynchronize());

	baropg_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
			d_rho, d_rmean, 
			kb, jm, im);

	
	/*
	checkCudaErrors(cudaMemcpy(drhox, d_drhox, kb*jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(drhoy, d_drhoy, kb*jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	*/
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_baropg);
		baropg_time += time_consumed(&start_baropg, 
									 &end_baropg);
#endif
	
    return;
}







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
					const int jm, const int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	const int i = blockDim.x*blockIdx.x + threadIdx.x; 
	const int jmm1 = jm-1; 
	const int imm1 = im-1; 
	float tps;

	float tmp_fluxua_advua; 
	float tmp_fluxva_advua; 
	float tmp_fluxua_advva; 
	float tmp_fluxva_advva; 

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			advua[j][i] = 0;	// for read and write restart file
		}
	}
	*/

	/*
	if (j < jm && i < im){ 
		advua[j_off+i] = 0;
	}
	*/
		
	/*
	for (j = 1; j < jm; j++){
		for (i = 1; i < imm1; i++){
			fluxua[j][i] = 0.125f*((d[j][i+1]+d[j][i])*ua[j][i+1]+(d[j][i]+d[j][i-1])*ua[j][i])*(ua[j][i+1]+ua[j][i]);
		}
	}
	*/

	/*
	if (j < jm && j > 0 && i < imm1 && i > 0){ 
	    fluxua[j_off+i] = 0.125f*((d[j_off+(i+1)]+d[j_off+i])
								    *ua[j_off+(i+1)]
								 +(d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i])
								*(ua[j_off+(i+1)]+ua[j_off+i]); 
	}
	*/

	/*
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = 0.125f*((d[j][i]+d[j-1][i])*va[j][i] + (d[j][i-1]+d[j-1][i-1])*va[j][i-1])*(ua[j][i] + ua[j-1][i]);
		}
	}
	*/
		
	/*
	if (j < jm && j > 0 && i < im && i > 0){ 
	    fluxva[j_off+i] = 0.125f*((d[j_off+i]+d[j_1_off+i])
									*va[j_off+i]
								 +(d[j_off+(i-1)]+d[j_1_off+(i-1)])
									*va[j_off+(i-1)])
								*(ua[j_off+i]+ua[j_1_off+i]); 
	} 
	*/

	/*
	for (j = 1; j < jm; j++){
		for (i = 1; i < imm1; i++){
			fluxua[j][i] = fluxua[j][i] - d[j][i]*2.0f*aam2d[j][i]*(uab[j][i+1]-uab[j][i])/dx[j][i];
		}
	}
	*/
		
	/*
	if (j < jm && j > 0 && i < imm1 && i > 0){ 
	    fluxua[j_off+i] = fluxua[j_off+i]
						 -d[j_off+i]*2.0f
						   *aam2d[j_off+i]
						   *(uab[j_off+(i+1)]
							-uab[j_off+i])
						   /dx[j_off+i];
	
	}
	*/
		
	/*
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			tps[j][i] = 0.25f*(d[j][i]+d[j][i-1]+d[j-1][i]+d[j-1][i-1])*(aam2d[j][i]+aam2d[j-1][i]+aam2d[j][i-1]+aam2d[j-1][i-1])*((uab[j][i]-uab[j-1][i])/(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1])+(vab[j][i]-vab[j][i-1])/(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1]));
			fluxua[j][i] *= dy[j][i];
			fluxva[j][i] = (fluxva[j][i]-tps[j][i])*0.25f*(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1]);
		}
	}
	*/
	/*
	if (j < jm && j > 0 && i < im && i > 0){ 
	    tps[j_off+i] = 0.25f*(d[j_off+i]+d[j_off+(i-1)]
							 +d[j_1_off+i]+d[j_1_off+(i-1)])
							*(aam2d[j_off+i]+aam2d[j_1_off+i]
							 +aam2d[j_off+(i-1)]+aam2d[j_1_off+(i-1)])
							*((uab[j_off+i]-uab[j_1_off+i])
							    /(dy[j_off+i]+dy[j_off+(i-1)]
							     +dy[j_1_off+i]+dy[j_1_off+(i-1)])
							 +(vab[j_off+i]-vab[j_off+(i-1)])
								/(dx[j_off+i]+dx[j_off+(i-1)]
								 +dx[j_1_off+i]+dx[j_1_off+(i-1)])); 

	   fluxua[j_off+i] *= dy[j_off+i]; 

	   fluxva[j_off+i] = (fluxva[j_off+i]-tps[j_off+i])
							*0.25f
							*(dx[j_off+i]+dx[j_off+(i-1)]
							 +dx[j_1_off+i]+dx[j_1_off+(i-1)]); 
	}
	*/

	if (j < jm && j > 0 && i < imm1 && i > 0){ 
	    tmp_fluxua_advua = 0.125f*((d[j_off+(i+1)]+d[j_off+i])
								    *ua[j_off+(i+1)]
								 +(d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i])
								*(ua[j_off+(i+1)]+ua[j_off+i]); 

	    tmp_fluxua_advua = tmp_fluxua_advua
						  -d[j_off+i]*2.0f
						    *aam2d[j_off+i]
						    *(uab[j_off+(i+1)]
						     -uab[j_off+i])
						    /dx[j_off+i];

		fluxua_advua[j_off+i] = tmp_fluxua_advua*dy[j_off+i]; 
	}

	if (j < jmm1 && j > 0 && i < im && i > 0){
		tmp_fluxva_advva = 0.125f*((d[j_A1_off+i]+d[j_off+i])
									*va[j_A1_off+i]
								 +(d[j_off+i]+d[j_1_off+i])
									*va[j_off+i])
								*(va[j_A1_off+i]+va[j_off+i]);

		tmp_fluxva_advva = tmp_fluxva_advva
						  -d[j_off+i]*2.0f
						 	*aam2d[j_off+i]
						 	*(vab[j_A1_off+i]-vab[j_off+i])
						 	/dy[j_off+i];

		fluxva_advva[j_off+i] = tmp_fluxva_advva*dx[j_off+i];
	}


	if (j < jm && j > 0 && i < im && i > 0){ 
	    tps = 0.25f*(d[j_off+i]+d[j_off+(i-1)]
		   		 +d[j_1_off+i]+d[j_1_off+(i-1)])
		   		*(aam2d[j_off+i]+aam2d[j_1_off+i]
		   		 +aam2d[j_off+(i-1)]+aam2d[j_1_off+(i-1)])
		   		*((uab[j_off+i]-uab[j_1_off+i])
		   		    /(dy[j_off+i]+dy[j_off+(i-1)]
		   		     +dy[j_1_off+i]+dy[j_1_off+(i-1)])
		   		 +(vab[j_off+i]-vab[j_off+(i-1)])
		   			/(dx[j_off+i]+dx[j_off+(i-1)]
		   			 +dx[j_1_off+i]+dx[j_1_off+(i-1)])); 

	    tmp_fluxva_advua = 0.125f*((d[j_off+i]+d[j_1_off+i])
									*va[j_off+i]
								 +(d[j_off+(i-1)]+d[j_1_off+(i-1)])
									*va[j_off+(i-1)])
								*(ua[j_off+i]+ua[j_1_off+i]); 

		fluxva_advua[j_off+i] = (tmp_fluxva_advua-tps)
							   *0.25f
							   *(dx[j_off+i]+dx[j_off+(i-1)]
							    +dx[j_1_off+i]+dx[j_1_off+(i-1)]); 

		tmp_fluxua_advva = 0.125f*((d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i]
								 +(d[j_1_off+i]+d[j_1_off+(i-1)])
									*ua[j_1_off+i])
								*(va[j_off+(i-1)]+va[j_off+i]);

		fluxua_advva[j_off+i] = (tmp_fluxua_advva-tps)
							*0.25f
							*(dy[j_off+i]+dy[j_off+(i-1)]
							 +dy[j_1_off+i]+dy[j_1_off+(i-1)]);
	}
}

__global__ void 
advave_inner_gpu_kernel_0(//float * __restrict__ advua, 
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
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const int jm, const int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	float tps;

	float tmp_fluxua_advua; 
	float tmp_fluxva_advua; 
	float tmp_fluxua_advva; 
	float tmp_fluxva_advva; 


	if (j > 32 && i > 32 && j < jm-33 && i < im-33){ 
	    tmp_fluxua_advua = 0.125f*((d[j_off+(i+1)]+d[j_off+i])
								    *ua[j_off+(i+1)]
								 +(d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i])
								*(ua[j_off+(i+1)]+ua[j_off+i]); 

	    tmp_fluxua_advua = tmp_fluxua_advua
						  -d[j_off+i]*2.0f
						    *aam2d[j_off+i]
						    *(uab[j_off+(i+1)]
						     -uab[j_off+i])
						    /dx[j_off+i];

		fluxua_advua[j_off+i] = tmp_fluxua_advua*dy[j_off+i]; 

		tmp_fluxva_advva = 0.125f*((d[j_A1_off+i]+d[j_off+i])
									*va[j_A1_off+i]
								 +(d[j_off+i]+d[j_1_off+i])
									*va[j_off+i])
								*(va[j_A1_off+i]+va[j_off+i]);

		tmp_fluxva_advva = tmp_fluxva_advva
						  -d[j_off+i]*2.0f
						 	*aam2d[j_off+i]
						 	*(vab[j_A1_off+i]-vab[j_off+i])
						 	/dy[j_off+i];

		fluxva_advva[j_off+i] = tmp_fluxva_advva*dx[j_off+i];
	}


	if (j < jm && i < im){ 
	    tps = 0.25f*(d[j_off+i]+d[j_off+(i-1)]
		   		 +d[j_1_off+i]+d[j_1_off+(i-1)])
		   		*(aam2d[j_off+i]+aam2d[j_1_off+i]
		   		 +aam2d[j_off+(i-1)]+aam2d[j_1_off+(i-1)])
		   		*((uab[j_off+i]-uab[j_1_off+i])
		   		    /(dy[j_off+i]+dy[j_off+(i-1)]
		   		     +dy[j_1_off+i]+dy[j_1_off+(i-1)])
		   		 +(vab[j_off+i]-vab[j_off+(i-1)])
		   			/(dx[j_off+i]+dx[j_off+(i-1)]
		   			 +dx[j_1_off+i]+dx[j_1_off+(i-1)])); 

	    tmp_fluxva_advua = 0.125f*((d[j_off+i]+d[j_1_off+i])
									*va[j_off+i]
								 +(d[j_off+(i-1)]+d[j_1_off+(i-1)])
									*va[j_off+(i-1)])
								*(ua[j_off+i]+ua[j_1_off+i]); 

		fluxva_advua[j_off+i] = (tmp_fluxva_advua-tps)
							   *0.25f
							   *(dx[j_off+i]+dx[j_off+(i-1)]
							    +dx[j_1_off+i]+dx[j_1_off+(i-1)]); 

		tmp_fluxua_advva = 0.125f*((d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i]
								 +(d[j_1_off+i]+d[j_1_off+(i-1)])
									*ua[j_1_off+i])
								*(va[j_off+(i-1)]+va[j_off+i]);

		fluxua_advva[j_off+i] = (tmp_fluxua_advva-tps)
							*0.25f
							*(dy[j_off+i]+dy[j_off+(i-1)]
							 +dy[j_1_off+i]+dy[j_1_off+(i-1)]);
	}
}


__global__ void 
advave_ew_gpu_kernel_0(
					const float * __restrict__ d, 
					const float * __restrict__ ua, 
					const float * __restrict__ va,  
					float * __restrict__ fluxua_advua, 
					float * __restrict__ fluxva_advva, 
					const float * __restrict__ uab,
					const float * __restrict__ vab, 
					const float * __restrict__ aam2d, 
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const int jm, const int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	float tmp_fluxua_advua; 
	float tmp_fluxva_advva; 

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	if (j < jm-1){ 
	    tmp_fluxua_advua = 0.125f*((d[j_off+(i+1)]+d[j_off+i])
								    *ua[j_off+(i+1)]
								 +(d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i])
								*(ua[j_off+(i+1)]+ua[j_off+i]); 

	    tmp_fluxua_advua = tmp_fluxua_advua
						  -d[j_off+i]*2.0f
						    *aam2d[j_off+i]
						    *(uab[j_off+(i+1)]
						     -uab[j_off+i])
						    /dx[j_off+i];

		fluxua_advua[j_off+i] = tmp_fluxua_advua*dy[j_off+i]; 

		tmp_fluxva_advva = 0.125f*((d[j_A1_off+i]+d[j_off+i])
									*va[j_A1_off+i]
								 +(d[j_off+i]+d[j_1_off+i])
									*va[j_off+i])
								*(va[j_A1_off+i]+va[j_off+i]);

		tmp_fluxva_advva = tmp_fluxva_advva
						  -d[j_off+i]*2.0f
						 	*aam2d[j_off+i]
						 	*(vab[j_A1_off+i]-vab[j_off+i])
						 	/dy[j_off+i];

		fluxva_advva[j_off+i] = tmp_fluxva_advva*dx[j_off+i];
	}
}


__global__ void 
advave_sn_gpu_kernel_0(
					const float * __restrict__ d, 
					const float * __restrict__ ua, 
					const float * __restrict__ va,  
					float * __restrict__ fluxua_advua, 
					float * __restrict__ fluxva_advva, 
					const float * __restrict__ uab,
					const float * __restrict__ vab, 
					const float * __restrict__ aam2d, 
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const int jm, const int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	float tmp_fluxua_advua; 
	float tmp_fluxva_advva; 

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){ 
	    tmp_fluxua_advua = 0.125f*((d[j_off+(i+1)]+d[j_off+i])
								    *ua[j_off+(i+1)]
								 +(d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i])
								*(ua[j_off+(i+1)]+ua[j_off+i]); 

	    tmp_fluxua_advua = tmp_fluxua_advua
						  -d[j_off+i]*2.0f
						    *aam2d[j_off+i]
						    *(uab[j_off+(i+1)]
						     -uab[j_off+i])
						    /dx[j_off+i];

		fluxua_advua[j_off+i] = tmp_fluxua_advua*dy[j_off+i]; 

		tmp_fluxva_advva = 0.125f*((d[j_A1_off+i]+d[j_off+i])
									*va[j_A1_off+i]
								 +(d[j_off+i]+d[j_1_off+i])
									*va[j_off+i])
								*(va[j_A1_off+i]+va[j_off+i]);

		tmp_fluxva_advva = tmp_fluxva_advva
						  -d[j_off+i]*2.0f
						 	*aam2d[j_off+i]
						 	*(vab[j_A1_off+i]-vab[j_off+i])
						 	/dy[j_off+i];

		fluxva_advva[j_off+i] = tmp_fluxva_advva*dx[j_off+i];
	}
}

__global__ void
advave_ew_bcond_gpu_kernel_0(float * __restrict__ fluxua_advua,
							int n_west,
							int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 

	if (n_west == -1){
		if (j < jm-1){
			fluxua_advua[j_off] = 0;
		}
	}


}

__global__ void
advave_sn_bcond_gpu_kernel_0(float* __restrict__ fluxva_advva,
							int n_south,
							int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 

	if (n_south == -1){
		if (i < im-1){
			fluxva_advva[i] = 0;
		}
	}

}

__global__ void 
advave_gpu_kernel_1(float * __restrict__ advua, 
					float * __restrict__ advva, 
			        const float * __restrict__ fluxua_advua, 
					const float * __restrict__ fluxva_advua, 
			        const float * __restrict__ fluxua_advva, 
					const float * __restrict__ fluxva_advva, 
				    int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	//int kbm1 = kb-1; 
	int jmm1 = jm-1; 
	int imm1 = im-1; 
	
	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			advua[j][i] = fluxua[j][i]-fluxua[j][i-1]+fluxva[j+1][i]-fluxva[j][i];
		}
	}
	*/

	if (j < jmm1 && j > 0 && i < imm1 && i > 0){ 
	    advua[j_off+i] = fluxua_advua[j_off+i]-fluxua_advua[j_off+(i-1)]
						+fluxva_advua[j_A1_off+i]-fluxva_advua[j_off+i];	

		advva[j_off+i] = fluxua_advva[j_off+(i+1)]-fluxua_advva[j_off+i]
					    +fluxva_advva[j_off+i]-fluxva_advva[j_1_off+i];
	} 
}

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
				    int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	//int kbm1 = kb-1; 
	int jmm1 = jm-1; 
	//int imm1 = im-1; 

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			advva[j][i] = 0;	
		}
	}
	*/

	if (j < jm && i < im){
		advva[j_off+i] = 0;
	}

	/*
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			fluxua[j][i] = 0.125f*((d[j][i]+d[j][i-1])*ua[j][i]+(d[j-1][i]+d[j-1][i-1])*ua[j-1][i])*(va[j][i-1]+va[j][i]);
		}
	}
	*/
	
	if (j < jm && j > 0 && i < im && i > 0){
		fluxua[j_off+i] = 0.125f*((d[j_off+i]+d[j_off+(i-1)])
									*ua[j_off+i]
								 +(d[j_1_off+i]+d[j_1_off+(i-1)])
									*ua[j_1_off+i])
								*(va[j_off+(i-1)]+va[j_off+i]);
	}
	
	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = 0.125f*((d[j+1][i]+d[j][i])*va[j+1][i] + (d[j][i]+d[j-1][i])*va[j][i])*(va[j+1][i]+va[j][i]);
		}
	}
	*/
	if (j < jmm1 && j > 0 && i < im && i > 0){
		fluxva[j_off+i] = 0.125f*((d[j_A1_off+i]+d[j_off+i])
									*va[j_A1_off+i]
								 +(d[j_off+i]+d[j_1_off+i])
									*va[j_off+i])
								*(va[j_A1_off+i]+va[j_off+i]);
	}

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = fluxva[j][i] - d[j][i]*2.0f*aam2d[j][i]*(vab[j+1][i]-vab[j][i])/dy[j][i];	
		}
	}
	*/
	if (j < jmm1 && j > 0 && i < im && i > 0){
		fluxva[j_off+i] = fluxva[j_off+i]
						 -d[j_off+i]*2.0f
							*aam2d[j_off+i]
							*(vab[j_A1_off+i]-vab[j_off+i])
							/dy[j_off+i];
	}
	
	/*
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = fluxva[j][i]*dx[j][i];	
			fluxua[j][i] = (fluxua[j][i]-tps[j][i])*0.25f*(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1]);
		}
	}
	*/
	
	if (j < jm && j > 0 && i < im && i > 0){
		fluxva[j_off+i] = fluxva[j_off+i]*dx[j_off+i];
		fluxua[j_off+i] = (fluxua[j_off+i]-tps[j_off+i])
							*0.25f
							*(dy[j_off+i]+dy[j_off+(i-1)]
							 +dy[j_1_off+i]+dy[j_1_off+(i-1)]);
	}
}

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
					int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	int jmm1 = jm-1; 
	int imm1 = im-1; 

	/*
	for (j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			advva[j][i] = fluxua[j][i+1]-fluxua[j][i]+fluxva[j][i]-fluxva[j-1][i];	
		}
	}
	*/
	
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		advva[j_off+i] = fluxua[j_off+(i+1)]-fluxua[j_off+i]
					    +fluxva[j_off+i]-fluxva[j_1_off+i];
	}
}

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
					int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 

	int jmm1 = jm-1; 
	int imm1 = im-1; 


	
	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			float tmp = 0.25f*(vab[j][i]+vab[j+1][i]+vab[j][i-1]+vab[j+1][i-1]);
			wubot[j][i] = -0.5f*(cbc[j][i]+cbc[j][i-1])*sqrtf(uab[j][i]*uab[j][i]+tmp*tmp)*uab[j][i];
		}
	}
	*/
	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		float tmp = 0.25f*(vab[j_off+i]
						  +vab[j_A1_off+i]
						  +vab[j_off+(i-1)]
						  +vab[j_A1_off+(i-1)]);

		wubot[j_off+i] = -0.5f*(cbc[j_off+i]
							   +cbc[j_off+(i-1)])
							  *sqrtf(uab[j_off+i]*uab[j_off+i]
									+tmp*tmp)
							  *uab[j_off+i];
	}
	*/
	
	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			float tmp = 0.25f*(uab[j][i]+uab[j][i+1]+uab[j-1][i]+uab[j-1][i+1]);
			wvbot[j][i] = -0.5f*(cbc[j][i]+cbc[j-1][i])*sqrtf(vab[j][i]*vab[j][i]+tmp*tmp)*vab[j][i];
			//wvbot[j][i] = -0.5f*(cbc[j][i]+cbc[j-1][i])*sqrtf(powf(vab[j][i],2)+powf(tmp,2))*vab[j][i];
			//printf("I come here!\n");
		}
	}
	*/

	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		float tmp = 0.25f*(uab[j_off+i]
						  +uab[j_off+(i+1)]
						  +uab[j_1_off+i]
						  +uab[j_1_off+(i+1)]);

		wvbot[j_off+i] = -0.5f*(cbc[j_off+i]
							   +cbc[j_1_off+i])
							  *sqrtf(vab[j_off+i]*vab[j_off+i]
									+tmp*tmp)
							  *vab[j_off+i];
	}
	*/

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			curv2d[j][i] = 0.25f*((va[j+1][i]+va[j][i])*(dy[j][i+1]-dy[j][i-1])-(ua[j][i+1]+ua[j][i])*(dx[j+1][i]-dx[j-1][i]))/(dx[j][i]*dy[j][i]);	//xsz
		}
	}
	*/
	
	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		curv2d[j_off+i] = 0.25f*((va[j_A1_off+i]
									+va[j_off+i])
								  *(dy[j_off+(i+1)]
									-dy[j_off+(i-1)])
								-(ua[j_off+(i+1)]
									+ua[j_off+i])
								  *(dx[j_A1_off+i]
									-dx[j_1_off+i]))
							   /(dx[j_off+i]*dy[j_off+i]);
	}
	*/

	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		float tmp_wubot = 0.25f*(vab[j_off+i]
						  +vab[j_A1_off+i]
						  +vab[j_off+(i-1)]
						  +vab[j_A1_off+(i-1)]);

		wubot[j_off+i] = -0.5f*(cbc[j_off+i]
							   +cbc[j_off+(i-1)])
							  *sqrtf(uab[j_off+i]*uab[j_off+i]
									+tmp_wubot*tmp_wubot)
							  *uab[j_off+i];

		float tmp_wvbot = 0.25f*(uab[j_off+i]
					            +uab[j_off+(i+1)]
					            +uab[j_1_off+i]
					            +uab[j_1_off+(i+1)]);

		wvbot[j_off+i] = -0.5f*(cbc[j_off+i]
							   +cbc[j_1_off+i])
							  *sqrtf(vab[j_off+i]*vab[j_off+i]
									+tmp_wvbot*tmp_wvbot)
							  *vab[j_off+i];

		curv2d[j_off+i] = 0.25f*((va[j_A1_off+i]
									+va[j_off+i])
								  *(dy[j_off+(i+1)]
									-dy[j_off+(i-1)])
								-(ua[j_off+(i+1)]
									+ua[j_off+i])
								  *(dx[j_A1_off+i]
									-dx[j_1_off+i]))
							   /(dx[j_off+i]*dy[j_off+i]);
	}
}

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
					int jm, int im){
				    
	//modify +advua
	int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int jmm1 = jm-1; 
	int imm1 = im-1; 

    /*
    for (j = 1; j < jmm1; j++){
    	if (n_west == -1){
    		for (i = 2; i < imm1; i++){
    			advua[j][i] = advua[j][i]-aru[j][i]*0.25f*(curv2d[j][i]*d[j][i]*(va[j+1][i]+va[j][i])+curv2d[j][i-1]*d[j][i-1]*(va[j+1][i-1]+va[j][i-1]));
    		}
    	}else{
    		for (i = 1; i < imm1; i++){
    			advua[j][i] = advua[j][i]-aru[j][i]*0.25f*(curv2d[j][i]*d[j][i]*(va[j+1][i]+va[j][i])+curv2d[j][i-1]*d[j][i-1]*(va[j+1][i-1]+va[j][i-1]));
    		}
    	}
    }
    */

    if (n_west == -1){
    	if (j < jmm1 && j > 0 && i < imm1 && i >1){
    		advua[j_off+i] = advua[j_off+i]
    						-aru[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(va[j_A1_off+i]
    									+va[j_off+i])
    							 +curv2d[j_off+(i-1)]
    								*d[j_off+(i-1)]
    								*(va[j_A1_off+(i-1)]
    									+va[j_off+(i-1)]));
    		}
    	}
    else{
    	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
    		advua[j_off+i] = advua[j_off+i]
    						-aru[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(va[j_A1_off+i]
    									+va[j_off+i])
    							 +curv2d[j_off+(i-1)]
    								*d[j_off+(i-1)]
    								*(va[j_A1_off+(i-1)]
    									+va[j_off+(i-1)]));
    	}
    }

    /*
    for (i = 1; i < imm1; i++){
    	if (n_south == -1){
    		for (j = 2; j < jmm1; j++){
    			advva[j][i] = advva[j][i]+arv[j][i]*0.25f*(curv2d[j][i]*d[j][i]*(ua[j][i+1]+ua[j][i])+curv2d[j-1][i]*d[j-1][i]*(ua[j-1][i+1]+ua[j-1][i]));
    		}
    	}else{
    		for (j = 1; j < jmm1; j++){
    			advva[j][i] = advva[j][i]+arv[j][i]*0.25f*(curv2d[j][i]*d[j][i]*(ua[j][i+1]+ua[j][i])+curv2d[j-1][i]*d[j-1][i]*(ua[j-1][i+1]+ua[j-1][i]));	
    		}
    	}
    }
    */
    
    if (n_south == -1){ 
    	if (i < imm1 && i > 0 && j < jmm1 && j > 1){
        	advva[j_off+i] = advva[j_off+i]
    						+arv[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(ua[j_off+(i+1)]
    									+ua[j_off+i])
    							 +curv2d[j_1_off+i]
    								*d[j_1_off+i]
    								*(ua[j_1_off+(i+1)]
    									+ua[j_1_off+i])); 
    	}
    } 
    else{
    	if (i < imm1 && i > 0 && j < jmm1 && j > 0){
        	advva[j_off+i] = advva[j_off+i]
    						+arv[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(ua[j_off+(i+1)]
    									+ua[j_off+i])
    							 +curv2d[j_1_off+i]
    								*d[j_1_off+i]
    								*(ua[j_1_off+(i+1)]
    									+ua[j_1_off+i])); 
    	}
    }

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			float tmpu = (ua[j][i+1]-ua[j][i])/dx[j][i];
			float tmpv = (va[j+1][i]-va[j][i])/dy[j][i];
			float tmpuv = 0.25f*(ua[j+1][i]+ua[j+1][i+1]
								 -ua[j-1][i]-ua[j-1][i+1])/dy[j][i]
					     +0.25f*(va[j][i+1]+va[j+1][i+1]
								 -va[j][i-1]-va[j+1][i-1])/dx[j][i];

			aam2d[j][i] = (horcon*dx[j][i]*dy[j][i]
							*sqrtf((tmpu*tmpu)+(tmpv*tmpv)
								   +0.5f*(tmpuv*tmpuv))
						   +aam_init)
						  *(1.f+aamfrz[j][i]);		//!lyo:channel:
		}
	}
	*/

	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		float tmpu = (ua[j_off+i+1]-ua[j_off+i])/dx[j_off+i];
		float tmpv = (va[j_A1_off+i]-va[j_off+i])/dy[j_off+i];

		float tmpuv = 0.25f*(ua[j_A1_off+i]+ua[j_A1_off+i+1]
							-ua[j_1_off+i]-ua[j_1_off+i+1])/dy[j_off+i]
					 +0.25f*(va[j_off+i+1]+va[j_A1_off+i+1]
							-va[j_off+i-1]-va[j_A1_off+i-1])/dx[j_off+i];

		aam2d[j_off+i] = (horcon*dx[j_off+i]*dy[j_off+i]
						   *sqrtf((tmpu*tmpu)+(tmpv*tmpv)
								 +0.5f*(tmpuv*tmpuv))
						  +aam_init)
						*(1.f+aamfrz[j_off+i]);

	}

	/*
    if (j < jmm1 && j > 0 && i < imm1 && i >1){
		if (n_west == -1){
    		advua[j_off+i] = advua[j_off+i]
    						-aru[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(va[j_A1_off+i]
    									+va[j_off+i])
    							 +curv2d[j_off+(i-1)]
    								*d[j_off+(i-1)]
    								*(va[j_A1_off+(i-1)]
    									+va[j_off+(i-1)]));
    	}else{
    		advua[j_off+i] = advua[j_off+i]
    						-aru[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(va[j_A1_off+i]
    									+va[j_off+i])
    							 +curv2d[j_off+(i-1)]
    								*d[j_off+(i-1)]
    								*(va[j_A1_off+(i-1)]
    									+va[j_off+(i-1)]));
    	}

		if (n_south == -1){ 
        	advva[j_off+i] = advva[j_off+i]
    						+arv[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(ua[j_off+(i+1)]
    									+ua[j_off+i])
    							 +curv2d[j_1_off+i]
    								*d[j_1_off+i]
    								*(ua[j_1_off+(i+1)]
    									+ua[j_1_off+i])); 
		}else{
        	advva[j_off+i] = advva[j_off+i]
    						+arv[j_off+i]*0.25f
    							*(curv2d[j_off+i]
    								*d[j_off+i]
    								*(ua[j_off+(i+1)]
    									+ua[j_off+i])
    							 +curv2d[j_1_off+i]
    								*d[j_1_off+i]
    								*(ua[j_1_off+(i+1)]
    									+ua[j_1_off+i])); 
    	}

		float tmpu = (ua[j_off+i+1]-ua[j_off+i])/dx[j_off+i];
		float tmpv = (va[j_A1_off+i]-va[j_off+i])/dy[j_off+i];

		float tmpuv = 0.25f*(ua[j_A1_off+i]+ua[j_A1_off+i+1]
							-ua[j_1_off+i]-ua[j_1_off+i+1])/dy[j_off+i]
					 +0.25f*(va[j_off+i+1]+va[j_A1_off+i+1]
							-va[j_off+i-1]-va[j_A1_off+i-1])/dx[j_off+i];

		aam2d[j_off+i] = (horcon*dx[j_off+i]*dy[j_off+i]
						   *sqrtf((tmpu*tmpu)+(tmpv*tmpv)
								 +0.5f*(tmpuv*tmpuv))
						  +aam_init)
						*(1.f+aamfrz[j_off+i]);
    }
	*/
}

/*
void advave_gpu(float advua[][i_size], float d[][i_size],
				float ua[][i_size], float va[][i_size],
				float fluxua[][i_size], float fluxva[][i_size],
				float uab[][i_size], float aam2d[][i_size],
				float vab[][i_size], float advva[][i_size],
				float wubot[][i_size], float wvbot[][i_size]){
*/

/*
void advave_gpu(float *d_advua, float *d_advva,
				float *d_fluxua, float *d_fluxva,
				float *d_wubot, float *d_wvbot,
				float *d_d, float *d_aam2d,
				float *d_ua, float *d_va,
				float *d_uab, float *d_vab){
*/
void advave_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_advave,
				   end_advave;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advave);
#endif
				
	//int i,j,k;

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	float *d_fluxua_advua = d_2d_tmp0;
	float *d_fluxua_advua_east = d_2d_tmp0_east;
	float *d_fluxua_advua_west = d_2d_tmp0_west;
	float *d_fluxua_advua_south = d_2d_tmp0_south;
	float *d_fluxua_advua_north = d_2d_tmp0_north;

	float *d_fluxva_advva = d_2d_tmp1;
	float *d_fluxva_advva_east = d_2d_tmp1_east;
	float *d_fluxva_advva_west = d_2d_tmp1_west;
	float *d_fluxva_advva_south = d_2d_tmp1_south;
	float *d_fluxva_advva_north = d_2d_tmp1_north;

	float *d_curv2d = d_2d_tmp2;
	float *d_fluxva_advua = d_2d_tmp3;
	float *d_fluxua_advva = d_2d_tmp4;

	//float tps[j_size][i_size];
    //float curv2d[j_size][i_size];

/*change: 
 *   need not copy in: advua/fluxua/fluxva/advva     wubot/wvbot/
 *   need copy in : none
 */

	/*
	checkCudaErrors(cudaMemcpy(d_advua, advua, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advva, advva, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fluxua, fluxua, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fluxva, fluxva, jm*im*sizeof(float), cudaMemcpyHostToDevice));

	
	checkCudaErrors(cudaMemcpy(d_d, d, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ua, ua, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_va, va, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uab, uab, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam2d, aam2d, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vab, vab, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	
	dim3 threadPerBlock_ew(32, 4);
	dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	dim3 threadPerBlock_ew_bcond(1, 128);
	dim3 blockPerGrid_ew_bcond(1, (j_size-2+127)/128);

	dim3 threadPerBlock_sn(32, 4);
	dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	dim3 threadPerBlock_sn_bcond(128, 1);
	dim3 blockPerGrid_sn_bcond((i_size-2+127)/128, 1);

	//advave_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		d_d, d_ua, d_va, 
	//		d_fluxua_advua, d_fluxva_advua,
	//		d_fluxua_advva, d_fluxva_advva,
	//	    d_uab, d_vab, d_aam2d, 
	//		d_dx, d_dy, jm, im);

    ////exchange2d_mpi_gpu(d_fluxua_advua, im, jm);
    ////exchange2d_mpi_gpu(d_fluxva_advva,im,jm);
    //exchange2d_cuda_aware_mpi(d_fluxua_advua, im, jm);
    //exchange2d_cuda_aware_mpi(d_fluxva_advva,im,jm);

	advave_ew_gpu_kernel_0<<<blockPerGrid_ew, threadPerBlock_ew,
						  0, stream[1]>>>(
			d_d, d_ua, d_va, 
			d_fluxua_advua, d_fluxva_advva,
		    d_uab, d_vab, d_aam2d, 
			d_dx, d_dy, jm, im);

	advave_sn_gpu_kernel_0<<<blockPerGrid_sn, threadPerBlock_sn,
						  0, stream[2]>>>(
			d_d, d_ua, d_va, 
			d_fluxua_advua, d_fluxva_advva,
		    d_uab, d_vab, d_aam2d, 
			d_dx, d_dy, jm, im);


	advave_inner_gpu_kernel_0<<<blockPerGrid, threadPerBlock,
						  0, stream[0]>>>(
			d_d, d_ua, d_va, 
			d_fluxua_advua, d_fluxva_advua,
			d_fluxua_advva, d_fluxva_advva,
		    d_uab, d_vab, d_aam2d, 
			d_dx, d_dy, jm, im);

	advave_ew_bcond_gpu_kernel_0<<<blockPerGrid_ew_bcond, 
								  threadPerBlock_ew_bcond,
								  0, stream[3]>>>(
			d_fluxua_advua, n_west, jm, im);

	advave_sn_bcond_gpu_kernel_0<<<blockPerGrid_sn_bcond, 
								  threadPerBlock_sn_bcond,
							      0, stream[4]>>>(
			d_fluxva_advva, n_south, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

    //exchange2d_mpi_gpu(d_fluxua_advua, im, jm);
    //exchange2d_mpi_gpu(d_fluxva_advva,im,jm);

    exchange2d_cudaUVA(d_fluxua_advua, 
					   d_fluxua_advua_east, d_fluxua_advua_west,
					   d_fluxua_advua_south, d_fluxua_advua_north,
					   stream[1], im, jm);

    exchange2d_cudaUVA(d_fluxva_advva,
					   d_fluxva_advva_east, d_fluxva_advva_west,
					   d_fluxva_advva_south, d_fluxva_advva_north,
					   stream[1], im,jm);

	//MPI_Barrier(pom_comm);
    //exchange2d_cuda_ipc(d_fluxua_advua, 
	//					d_fluxua_advua_east, d_fluxua_advua_west,
	//					stream[1], im, jm);

    //exchange2d_cuda_ipc(d_fluxva_advva,
	//					d_fluxva_advva_east, d_fluxva_advva_west,
	//					stream[1], im,jm);
	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[3]));
	checkCudaErrors(cudaStreamSynchronize(stream[4]));
	checkCudaErrors(cudaStreamSynchronize(stream[0]));


	advave_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_advua, d_advva, 
			d_fluxua_advua, d_fluxva_advua,
			d_fluxua_advva, d_fluxva_advva,
			jm, im);

/*
#ifdef CUDA_SLICE_MPI
    exchange2d_mpi_gpu(d_fluxua, im, jm);
#else
	float fluxua[j_size][i_size];
	checkCudaErrors(cudaMemcpy(fluxua, d_fluxua, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));

    //exchange2d_mpi_xsz_(fluxua,im,jm);
    exchange2d_mpi(fluxua,im,jm);

	checkCudaErrors(cudaMemcpy(d_fluxua, fluxua, jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
#endif
*/
    //exchange2d_mpi_gpu(d_fluxua, im, jm);


	/*
	advave_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_advua, d_advva, d_fluxua, d_fluxva, 
			d_ua, d_va, d_d, d_aam2d, d_vab, d_tps, 
			d_dx, d_dy, jm, im);
	*/
	//advave_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
	//		d_advua, d_fluxua, d_fluxva,
	//		jm, im);


	//advave_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
	//		d_advva, d_fluxua, d_fluxva, 
	//		d_ua, d_va, d_d, d_aam2d, d_vab, d_tps, 
	//		d_dx, d_dy, jm, im);

/*
#ifdef CUDA_SLICE_MPI
    exchange2d_mpi_gpu(d_fluxva,im,jm);
#else
	float fluxva[j_size][i_size];
	checkCudaErrors(cudaMemcpy(fluxva, d_fluxva, jm*im*sizeof(float), 
				cudaMemcpyDeviceToHost));

    //exchange2d_mpi_xsz_(fluxva,im,jm);
    exchange2d_mpi(fluxva,im,jm);

	checkCudaErrors(cudaMemcpy(d_fluxva, fluxva, jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
#endif
*/

    //exchange2d_mpi_gpu(d_fluxva,im,jm);

	//advave_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
	//		d_advva, d_fluxua, d_fluxva, 
	//		jm, im);
	
	//checkCudaErrors(cudaDeviceSynchronize());

	if (mode == 2){

		advave_gpu_kernel_4<<<blockPerGrid, threadPerBlock>>>(
				d_wubot, d_wvbot, d_curv2d, d_uab ,d_vab, 
				d_cbc, d_ua, d_va, d_dx, d_dy, 
				jm, im);
	
/*
#ifdef CUDA_SLICE_MPI
        exchange2d_mpi_gpu(d_wubot,im,jm);
        exchange2d_mpi_gpu(d_wvbot,im,jm);
        exchange2d_mpi_gpu(d_curv2d,im,jm);
#else
		float curv2d[j_size][i_size];
		checkCudaErrors(cudaMemcpy(wubot, d_wubot, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(curv2d, d_curv2d, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

        //exchange2d_mpi_xsz_(curv2d,im,jm);
        exchange2d_mpi(wubot,im,jm);
        exchange2d_mpi(wvbot,im,jm);
        exchange2d_mpi(curv2d,im,jm);

		checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_curv2d, curv2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
#endif
*/
        //exchange2d_mpi_gpu(d_wubot,im,jm);
        //exchange2d_mpi_gpu(d_wvbot,im,jm);
        //exchange2d_mpi_gpu(d_curv2d,im,jm);
        exchange2d_cuda_aware_mpi(d_wubot,im,jm);
        exchange2d_cuda_aware_mpi(d_wvbot,im,jm);
        exchange2d_cuda_aware_mpi(d_curv2d,im,jm);


		advave_gpu_kernel_5<<<blockPerGrid, threadPerBlock>>>(
				d_advua, d_advva, d_d, d_ua, d_va, 
				d_aam2d, d_curv2d, d_aamfrz,
			    d_aru, d_arv, d_dx, d_dy, 
				horcon, aam_init, n_west, n_south, jm, im);
	
		//checkCudaErrors(cudaDeviceSynchronize());
	}

	//modify +advua + advva
	
	/*
	checkCudaErrors(cudaMemcpy(advua, d_advua, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(advva, d_advva, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fluxua, d_fluxua, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fluxva, d_fluxva, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(wubot, d_wubot, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advave);
		advave_time += time_consumed(&start_advave, 
									 &end_advave);
#endif

	return;
}



__global__ void
vertvl_gpu_kernel_0(float * __restrict__ xflux, 
					float * __restrict__ yflux, 
				    const float * __restrict__ u, 
					const float * __restrict__ v, 
					const float * __restrict__ dt,
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
				    int kb, int jm, int im){

	//only modified -w 
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;

	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				xflux[k][j][i] = 0.25f*(dy[j][i]+dy[j][i-1])
								  *(dt[j][i]+dt[j][i-1])*u[k][j][i];
			}
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			xflux[k_off+j_off+i] = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)])
										*(dt[j_off+i]+dt[j_off+(i-1)])
										*u[k_off+j_off+i];
		}
	}

	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				yflux[k][j][i] = 0.25f*(dx[j][i]+dx[j-1][i])
								  *(dt[j][i]+dt[j-1][i])*v[k][j][i];
			}
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			yflux[k_off+j_off+i] = 0.25f*(dx[j_off+i]+dx[j_1_off+i])
									    *(dt[j_off+i]+dt[j_1_off+i])
										*v[k_off+j_off+i];
		}
	}

}

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
				    float dti2, int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			w[0][j][i] = 0.5f*(vfluxb[j][i]+vfluxf[j][i]);
		}
	}
	*/
	
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		w[j_off+i] = 0.5f*(vfluxb[j_off+i]+vfluxf[j_off+i]);
	}

	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				w[k+1][j][i] = w[k][j][i]+dz[k]*((xflux[k][j][i+1]-xflux[k][j][i]
									+yflux[k][j+1][i]-yflux[k][j][i])
								  /(dx[j][i]*dy[j][i])
							    +(etf[j][i]-etb[j][i])/dti2);
			}
		}
	}
	*/
	
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 0; k < kbm1; k++){
			w[k_A1_off+j_off+i] = w[k_off+j_off+i]
								 +dz[k]*((xflux[k_off+j_off+(i+1)]
										   -xflux[k_off+j_off+i]
										   +yflux[k_off+j_A1_off+i]
										   -yflux[k_off+j_off+i]) 
										 /(dx[j_off+i]*dy[j_off+i]) 
										+(etf[j_off+i]-etb[j_off+i])/dti2);	
		}
	}

}
/*
void vertvl_gpu(float dt[][i_size], float u[][j_size][i_size],
				float v[][j_size][i_size],float vfluxb[][i_size],
				float vfluxf[][i_size], float w[][j_size][i_size],
				float etf[][i_size],float etb[][i_size]){
*/

/*
void vertvl_gpu(float *d_u, float *d_v, float *d_w,
				float *d_vfluxb, float *d_vfluxf, 
				float *d_etf,float *d_etb,
				float *d_dt){
*/
void vertvl_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_vertvl,
				   end_vertvl;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_vertvl);
#endif

	//modified -w 
	//NO! the boundary is useful!, so we need copy-in w
	//int i,j,k;

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	/*
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	*/

    float *d_xflux = d_3d_tmp0;
	float *d_yflux = d_3d_tmp1;
	
	/*
	checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_vfluxb, vfluxb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfluxf, vfluxf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	

	//only modified -w 
	
	/*
	vertvl_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xflux, d_yflux, d_u, d_v, d_w,
			d_vfluxb, d_vfluxf, d_etf, d_etb, d_dt,
		    d_dx, d_dy, d_dz, dti2, kb, jm, im);
	*/
	
	vertvl_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xflux, d_yflux, d_u, d_v, d_dt,
			d_dx, d_dy, kb, jm, im);

	vertvl_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_w, d_vfluxb, d_vfluxf,
		    d_xflux, d_yflux, d_etf, d_etb, 
			d_dx, d_dy, d_dz,
			dti2, kb, jm, im);

	//checkCudaErrors(cudaMemcpy(w, d_w, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaDeviceSynchronize());

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_vertvl);
		vertvl_time += time_consumed(&start_vertvl, 
									 &end_vertvl);
#endif

    return;
}



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
				  int kb, int jm, int im){

	//modify -xflux, yflux
#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
	float xflux_tmp = 0, yflux_tmp = 0;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	//int jmm1 = jm-1;
	//int imm1 = im-1;


#ifdef D3_BLOCK
	if (k > 0 && k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
		xflux_tmp = 0.125f*(q[k_off+j_off+i]
							+q[k_off+j_off+(i-1)])
						  *(dt[j_off+i]
							+dt[j_off+(i-1)])
						  *(u[k_off+j_off+i]
							+u[k_1_off+j_off+i]);	

		yflux_tmp = 0.125f*(q[k_off+j_off+i]
							+q[k_off+j_1_off+i])
						  *(dt[j_off+i]
							+dt[j_1_off+i])
						  *(v[k_off+j_off+i]
							+v[k_1_off+j_off+i]);
	}


	if (k > 0 && k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
		xflux_tmp = xflux_tmp
				   -0.25f*(aam[k_off+j_off+i]
						  +aam[k_off+j_off+(i-1)]
						  +aam[k_1_off+j_off+i]
						  +aam[k_1_off+j_off+(i-1)])
						 *(h[j_off+i]+h[j_off+(i-1)])
						 *(qb[k_off+j_off+i]
						  -qb[k_off+j_off+(i-1)])
						 *dum[j_off+i]
						 /(dx[j_off+i]+dx[j_off+(i-1)]);	

		xflux[k_off+j_off+i] = 0.5f*(dy[j_off+i]
									+dy[j_off+(i-1)])
								   *xflux_tmp;

		yflux_tmp = yflux_tmp
				   -0.25f*(aam[k_off+j_off+i]
						  +aam[k_off+j_1_off+i]
						  +aam[k_1_off+j_off+i]
						  +aam[k_1_off+j_1_off+i])
						 *(h[j_off+i]+h[j_1_off+i])
						 *(qb[k_off+j_off+i]
						  -qb[k_off+j_1_off+i])
						 *dvm[j_off+i]
						 /(dy[j_off+i]+dy[j_1_off+i]);

		yflux[k_off+j_off+i] = 0.5f*(dx[j_off+i]
									+dx[j_1_off+i])
								   *yflux_tmp;
	}
#else

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				xflux[k][j][i] = 0;
				yflux[k][j][i] = 0;
			}
		}
	}
	*/

	/*
	if (k < kb && j < jm && i < im){
		xflux[k_off+j_off+i] = 0;
		yflux[k_off+j_off+i] = 0;
	}
	*/

	/*
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			xflux[k_off+j_off+i] = 0;
			yflux[k_off+j_off+i] = 0;
		}
	}
	*/

//! do horizontal advection

	/*
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i] = 0.125f*(q[k][j][i]+q[k][j][i-1])*(dt[j][i]+dt[j][i-1])*(u[k][j][i]+u[k-1][j][i]);
				yflux[k][j][i] = 0.125f*(q[k][j][i]+q[k][j-1][i])*(dt[j][i]+dt[j-1][i])*(v[k][j][i]+v[k-1][j][i]);
			}
		}
	}
	*/

	/*
	if (k > 0 && k < kbm1 && j > 0 && j < jm && i > 0 && i < im){

		xflux[k_off+j_off+i] = 0.125f*(q[k_off+j_off+i]+q[k_off+j_off+(i-1)])*(dt[j_off+i]+dt[j_off+(i-1)])*(u[k_off+j_off+i]+u[k_1_off+j_off+i]);	
		yflux[k_off+j_off+i] = 0.125f*(q[k_off+j_off+i]+q[k_off+j_1_off+i])*(dt[j_off+i]+dt[j_1_off+i])*(v[k_off+j_off+i]+v[k_1_off+j_off+i]);
	}
	*/

	/*
	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jm && i > 0 && i < im){
			xflux[k_off+j_off+i] = 0.125f*(q[k_off+j_off+i]+q[k_off+j_off+(i-1)])*(dt[j_off+i]+dt[j_off+(i-1)])*(u[k_off+j_off+i]+u[k_1_off+j_off+i]);	
			yflux[k_off+j_off+i] = 0.125f*(q[k_off+j_off+i]+q[k_off+j_1_off+i])*(dt[j_off+i]+dt[j_1_off+i])*(v[k_off+j_off+i]+v[k_1_off+j_off+i]);
		}
	}
	*/

	/*
	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 1; k < kbm1; k++){
			xflux_tmp[k] = 0.125f*(q[k_off+j_off+i]
								  +q[k_off+j_off+(i-1)])
								 *(dt[j_off+i]
								  +dt[j_off+(i-1)])
								 *(u[k_off+j_off+i]
								  +u[k_1_off+j_off+i]);	

			yflux_tmp[k] = 0.125f*(q[k_off+j_off+i]
								  +q[k_off+j_1_off+i])
								 *(dt[j_off+i]
								  +dt[j_1_off+i])
								 *(v[k_off+j_off+i]
								  +v[k_1_off+j_off+i]);
		}
	}
	*/

//! do horizontal diffusion
	/*
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i]=xflux[k][j][i]-0.25f*(aam[k][j][i]+aam[k][j][i-1]+aam[k-1][j][i]+aam[k-1][j][i-1])*(h[j][i]+h[j][i-1])*(qb[k][j][i]-qb[k][j][i-1])*dum[j][i]/(dx[j][i]+dx[j][i-1]);
				yflux[k][j][i]=yflux[k][j][i]-0.25f*(aam[k][j][i]+aam[k][j-1][i]+aam[k-1][j][i]+aam[k-1][j-1][i])*(h[j][i]+h[j-1][i])*(qb[k][j][i]-qb[k][j-1][i])*dvm[j][i]/(dy[j][i]+dy[j-1][i]);
				xflux[k][j][i]=0.5f*(dy[j][i]+dy[j][i-1])*xflux[k][j][i];
				yflux[k][j][i]=0.5f*(dx[j][i]+dx[j-1][i])*yflux[k][j][i];
			}
		}
	}
	*/

	/*
	if (k > 0 && k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
		xflux[k_off+j_off+i] = xflux[k_off+j_off+i]-0.25f*(aam[k_off+j_off+i]+aam[k_off+j_off+(i-1)]+aam[k_1_off+j_off+i]+aam[k_1_off+j_off+(i-1)])*
														  (h[j_off+i]+h[j_off+(i-1)])*(qb[k_off+j_off+i]-qb[k_off+j_off+(i-1)])*dum[j_off+i]/(dx[j_off+i]+dx[j_off+(i-1)]);	
		xflux[k_off+j_off+i] = 0.5f*(dy[j_off+i]+dy[j_off+(i-1)])*xflux[k_off+j_off+i];

		yflux[k_off+j_off+i] = yflux[k_off+j_off+i]-0.25f*(aam[k_off+j_off+i]+aam[k_off+j_1_off+i]+aam[k_1_off+j_off+i]+aam[k_1_off+j_1_off+i])*
														  (h[j_off+i]+h[j_1_off+i])*(qb[k_off+j_off+i]-qb[k_off+j_1_off+i])*dvm[j_off+i]/
														  (dy[j_off+i]+dy[j_1_off+i]);
		yflux[k_off+j_off+i] = 0.5f*(dx[j_off+i]+dx[j_1_off+i])*yflux[k_off+j_off+i];
	}
	*/

	/*
	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 1; k < kbm1; k++){
			xflux_tmp[k] = xflux_tmp[k]
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_off+(i-1)]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_off+(i-1)])
								*(h[j_off+i]+h[j_off+(i-1)])
								*(qb[k_off+j_off+i]
								 -qb[k_off+j_off+(i-1)])
								*dum[j_off+i]
								/(dx[j_off+i]+dx[j_off+(i-1)]);	

			xflux[k_off+j_off+i] = 0.5f*(dy[j_off+i]
										+dy[j_off+(i-1)])
									   *xflux_tmp[k];

			yflux_tmp[k] = yflux_tmp[k]
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_1_off+i]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_1_off+i])
								*(h[j_off+i]+h[j_1_off+i])
								*(qb[k_off+j_off+i]
								 -qb[k_off+j_1_off+i])
								*dvm[j_off+i]
								/(dy[j_off+i]+dy[j_1_off+i]);

			yflux[k_off+j_off+i] = 0.5f*(dx[j_off+i]
										+dx[j_1_off+i])
									   *yflux_tmp[k];
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 1; k < kbm1; k++){
			float xflux_tmp,  yflux_tmp;
			xflux_tmp = 0.125f*(q[k_off+j_off+i]
								  +q[k_off+j_off+(i-1)])
								 *(dt[j_off+i]
								  +dt[j_off+(i-1)])
								 *(u[k_off+j_off+i]
								  +u[k_1_off+j_off+i]);	

			yflux_tmp = 0.125f*(q[k_off+j_off+i]
								  +q[k_off+j_1_off+i])
								 *(dt[j_off+i]
								  +dt[j_1_off+i])
								 *(v[k_off+j_off+i]
								  +v[k_1_off+j_off+i]);

			xflux_tmp = xflux_tmp
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_off+(i-1)]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_off+(i-1)])
								*(h[j_off+i]+h[j_off+(i-1)])
								*(qb[k_off+j_off+i]
								 -qb[k_off+j_off+(i-1)])
								*dum[j_off+i]
								/(dx[j_off+i]+dx[j_off+(i-1)]);	


			yflux_tmp = yflux_tmp
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_1_off+i]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_1_off+i])
								*(h[j_off+i]+h[j_1_off+i])
								*(qb[k_off+j_off+i]
								 -qb[k_off+j_1_off+i])
								*dvm[j_off+i]
								/(dy[j_off+i]+dy[j_1_off+i]);

			xflux[k_off+j_off+i] = 0.5f*(dy[j_off+i]
										+dy[j_off+(i-1)])
									   *xflux_tmp;

			yflux[k_off+j_off+i] = 0.5f*(dx[j_off+i]
										+dx[j_1_off+i])
									   *yflux_tmp;
		}
	}
#endif

}

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
				  float dti2, int kb, int jm, int im){

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

#ifdef D3_BLOCK
	if (k > 0 && k < kbm1
			&& j > 0 && j < jmm1 && i > 0 && i < imm1){
		float tmp;
		tmp = (w[k_1_off+j_off+i]
					*q[k_1_off+j_off+i]
				-w[k_A1_off+j_off+i]
					*q[k_A1_off+j_off+i])
			   *art[j_off+i]/(dz[k]+dz[k-1])
			 +xflux[k_off+j_off+(i+1)]
			 -xflux[k_off+j_off+i]
			 +yflux[k_off+j_A1_off+i]
			 -yflux[k_off+j_off+i];	

		qf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i])
								*art[j_off+i]*qb[k_off+j_off+i]
							  -dti2*tmp)
							/((h[j_off+i]+etf[j_off+i])
								*art[j_off+i]);
	}
#else
	/*
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				qf[k][j][i]=(w[k-1][j][i]*q[k-1][j][i]-w[k+1][j][i]*q[k+1][j][i])*art[j][i]/(dz[k]+dz[k-1])+xflux[k][j][i+1]-xflux[k][j][i]+yflux[k][j+1][i]-yflux[k][j][i];
				qf[k][j][i]=((h[j][i]+etb[j][i])*art[j][i]*qb[k][j][i]-dti2*qf[k][j][i])/((h[j][i]+etf[j][i])*art[j][i]);
			}
		}
	}
	*/

	/*
	if (k > 0 && k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		qf[k_off+j_off+i] = (w[k_1_off+j_off+i]*q[k_1_off+j_off+i] - w[k_A1_off+j_off+i]*q[k_A1_off+j_off+i])*art[j_off+i]/(dz[k]+dz[k-1]) +
							xflux[k_off+j_off+(i+1)]-xflux[k_off+j_off+i]+yflux[k_off+j_A1_off+i]-yflux[k_off+j_off+i];	
		qf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i])*art[j_off+i]*qb[k_off+j_off+i] - dti2*qf[k_off+j_off+i])/
							((h[j_off+i]+etf[j_off+i])*art[j_off+i]);
	}
	*/

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 1; k < kbm1; k++){
			float tmp;
			tmp = (w[k_1_off+j_off+i]
						*q[k_1_off+j_off+i]
					-w[k_A1_off+j_off+i]
						*q[k_A1_off+j_off+i])
				   *art[j_off+i]/(dz[k]+dz[k-1])
				  +xflux[k_off+j_off+(i+1)]
				  -xflux[k_off+j_off+i]
				  +yflux[k_off+j_A1_off+i]
				  -yflux[k_off+j_off+i];	

			qf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i])
									*art[j_off+i]*qb[k_off+j_off+i]
								  -dti2*tmp)
								/((h[j_off+i]+etf[j_off+i])
									*art[j_off+i]);
		}
	}
#endif
}

/*
void advq_gpu(float qb[][j_size][i_size], float q[][j_size][i_size],
			  float qf[][j_size][i_size], float u[][j_size][i_size],
			  float dt[][i_size], float v[][j_size][i_size],
			  float aam[][j_size][i_size], float w[][j_size][i_size],
			  float etb[][i_size], float etf[][i_size]){
*/

/*
void advq_gpu(float *d_qb, float *d_q, float *d_qf, 
			  float *d_u, float *d_v, float *d_w,
			  float *d_etb, float *d_etf,
			  float *d_aam, float *d_dt){
*/
void advq_gpu(float *d_qb, float *d_q, float *d_qf){ 

#ifndef TIME_DISABLE
	struct timeval start_advq,
				   end_advq;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advq);
#endif

	//modify:
	//     -qf
	//int i,j,k;
	/*
	float *d_qb = d_qb_advq;
	float *d_q = d_q_advq;
	float *d_qf = d_qf_advq;
	*/

	float *d_xflux = d_3d_tmp0;
	float *d_yflux = d_3d_tmp1;

    //float xflux[k_size][j_size][i_size];
	//float yflux[k_size][j_size][i_size];

#ifdef D3_BLOCK
	dim3 threadPerBlock(block_i_3D, block_j_3D, block_k_3D);
	dim3 blockPerGrid((im+block_i_3D-1)/block_i_3D, (jm+block_j_3D-1)/block_j_3D, (kb+block_k_3D-1)/block_k_3D);
	dim3 threadPerBlock_3x2(block_i_3D, block_k_3D, block_j_3D);
	dim3 blockPerGrid_3x2((im+block_i_3D-1)/block_i_3D, (jm+block_k_3D-1)/block_k_3D, (kb+block_j_3D-1)/block_j_3D);
#else
	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((im+block_i_2D-1)/block_i_2D, (jm+block_j_2D-1)/block_j_2D);
#endif
	
	/*
	checkCudaErrors(cudaMemcpy(d_qb, qb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q, q, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	

//! do horizontal advection

//! do horizontal diffusion

	//modify -xflux, yflux

	/*
	advq_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xflux_advq, d_yflux_advq, d_q, d_dt, d_u, d_v, d_aam, d_qb, 
			d_dum, d_dvm, d_h, d_dx, d_dy, kb, jm, im);
	*/

	advq_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xflux, d_yflux, d_q, d_qb,
			d_u, d_v, d_aam, d_dt, 
			d_dum, d_dvm, d_h, d_dx, d_dy, kb, jm, im);

/*
#ifdef CUDA_SLICE_MPI
    exchange3d_mpi_gpu(d_xflux,im,jm,kbm1);
    exchange3d_mpi_gpu(d_yflux,im,jm,kbm1);
#else
    float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	checkCudaErrors(cudaMemcpy(xflux, d_xflux, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(yflux, d_yflux, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));

    exchange3d_mpi(xflux,im,jm,kbm1);
    exchange3d_mpi(yflux,im,jm,kbm1);

	checkCudaErrors(cudaMemcpy(d_xflux, xflux, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_yflux, yflux, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
#endif
*/
    //exchange3d_mpi_gpu(d_xflux,im,jm,kbm1);
    //exchange3d_mpi_gpu(d_yflux,im,jm,kbm1);

    //exchange3d_cuda_aware_mpi(d_xflux,im,jm,kbm1);
    //exchange3d_cuda_aware_mpi(d_yflux,im,jm,kbm1);

//! do vertical advection, add flux terms, then step forward in time
	//modify -qf
	/*
	advq_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_qf, d_qb, d_q, d_w, d_xflux_advq, d_yflux_advq, 
			d_etb, d_etf, d_art, d_dz, d_h, dti2, kb, jm, im);
	*/

#ifdef D3_BLOCK
	advq_gpu_kernel_1<<<blockPerGrid_3x2, threadPerBlock_3x2>>>(
			d_qf, d_qb, d_q, d_w, 
			d_xflux, d_yflux, d_etb, d_etf, 
			d_art, d_dz, d_h, dti2, kb, jm, im);
#else
	advq_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_qf, d_qb, d_q, d_w, 
			d_xflux, d_yflux, d_etb, d_etf, 
			d_art, d_dz, d_h, dti2, kb, jm, im);
#endif
	
	//checkCudaErrors(cudaMemcpy(qf, d_qf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advq);
		advq_time += time_consumed(&start_advq, 
								   &end_advq);
#endif

	return;

}


__global__ void
advt1_gpu_kernel_0(float * __restrict__ f, 
				   float * __restrict__ fb, 
				   const float * __restrict__ fclim, 
				   int kb, int jm, int im){

//comments:
//     in this section ,we assign f&fb new value on its boundary(k = kbm1), 
//     but we don't use the boundary values
	//int k = blockDim.z*blockIdx.z + threadIdx.z;

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f[kb-1][j][i] = f[kbm1-1][j][i];
			fb[kb-1][j][i] = fb[kbm1-1][j][i];
		}
	}
	*/

	/*
	if (j < jm && i < im && k == 0){
		f[kb_1_off+j_off+i] = f[kbm1_1_off+j_off+i];
		fb[kb_1_off+j_off+i] = fb[kbm1_1_off+j_off+i];
	}
	*/

	if (j < jm && i < im){
		f[kb_1_off+j_off+i] = f[kbm1_1_off+j_off+i];
		fb[kb_1_off+j_off+i] = fb[kbm1_1_off+j_off+i];
	}


	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] = fb[k][j][i]-fclim[k][j][i];	
			}
		}
	}
	*/

	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			fb[k_off+j_off+i] -= fclim[k_off+j_off+i];	
		}
	}
}

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
				   float tprni, int kb, int jm, int im){

//comments:
//     in this section ,we assign f&fb new value on its boundary(k = kbm1), 
//     but we don't use the boundary values
	//int k = blockDim.z*blockIdx.z + threadIdx.z;

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i]=0.25f*((dt[j][i]+dt[j][i-1])*(f[k][j][i]+f[k][j][i-1])*u[k][j][i]);	
				yflux[k][j][i]=0.25f*((dt[j][i]+dt[j-1][i])*(f[k][j][i]+f[k][j-1][i])*v[k][j][i]);
			}
		}
	}
	*/
	
	/*
	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
		xflux[k_off+j_off+i] = 0.25f*((dt[j_off+i]+dt[j_off+(i-1)])*(f[k_off+j_off+i]+f[k_off+j_off+(i-1)])*u[k_off+j_off+i]);
		yflux[k_off+j_off+i] = 0.25f*((dt[j_off+i]+dt[j_1_off+i])*(f[k_off+j_off+i]+f[k_off+j_1_off+i])*v[k_off+j_off+i]);
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jm && i > 0 && i < im){
			xflux[k_off+j_off+i] = 0.25f*((dt[j_off+i]+dt[j_off+(i-1)])
										 *(f[k_off+j_off+i]
											+f[k_off+j_off+(i-1)])
										 *u[k_off+j_off+i]);
			yflux[k_off+j_off+i] = 0.25f*((dt[j_off+i]+dt[j_1_off+i])
									    *(f[k_off+j_off+i]
											+f[k_off+j_1_off+i])
										*v[k_off+j_off+i]);
		}
	}
	*/
	

	//! do advective fluxes

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i]=xflux[k][j][i]-0.5f*(aam[k][j][i]+aam[k][j][i-1])*(h[j][i]+h[j][i-1])*tprni*(fb[k][j][i]-fb[k][j][i-1])*dum[j][i]/(dx[j][i]+dx[j][i-1]);
				yflux[k][j][i]=yflux[k][j][i]-0.5f*(aam[k][j][i]+aam[k][j-1][i])*(h[j][i]+h[j-1][i])*tprni*(fb[k][j][i]-fb[k][j-1][i])*dvm[j][i]/(dy[j][i]+dy[j-1][i]);
				xflux[k][j][i]=0.5f*(dy[j][i]+dy[j][i-1])*xflux[k][j][i];
				yflux[k][j][i]=0.5f*(dx[j][i]+dx[j-1][i])*yflux[k][j][i];

			}
		}
	}
	*/
	
	/*
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jm && i > 0 && i < im){
			xflux[k_off+j_off+i] -= 0.5f*(aam[k_off+j_off+i]
											+aam[k_off+j_off+(i-1)])
										*(h[j_off+i]+h[j_off+(i-1)])
										*tprni
										*(fb[k_off+j_off+i]
										 -fb[k_off+j_off+(i-1)])
										*dum[j_off+i]
										/(dx[j_off+i]+dx[j_off+(i-1)]);	

			xflux[k_off+j_off+i] = 0.5f*(dy[j_off+i]
											+dy[j_off+(i-1)])
									   *xflux[k_off+j_off+i];

			yflux[k_off+j_off+i] -= 0.5f*(aam[k_off+j_off+i]
										    +aam[k_off+j_1_off+i])
										*(h[j_off+i]+h[j_1_off+i])
										*tprni
										*(fb[k_off+j_off+i]
										 -fb[k_off+j_1_off+i])
										*dvm[j_off+i]
										/(dy[j_off+i]+dy[j_1_off+i]);

			yflux[k_off+j_off+i] = 0.5f*(dx[j_off+i]
											+dx[j_1_off+i])
									   *yflux[k_off+j_off+i];
		}
	}
	*/

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			if (var == 'T')
				zflux[0][j][i] = tsurf[j][i]*w[0][j][i]*art[j][i];	
			else if (var == 'S')
				zflux[0][j][i] = 0;

			zflux[kb-1][j][i]=0;
		}
	}
	*/
	
	/*
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		if (var == 'T')
			zflux[j_off+i] = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
		else if (var == 'S')
			zflux[j_off+i] = 0;
		zflux[kb_1_off+j_off+i] = 0;
	}
	*/

	/*
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				zflux[k][j][i]=0.5f*(f[k-1][j][i]+f[k][j][i])*w[k][j][i]*art[j][i];	
			}
		}
	}
	*/
	
	/*
	if (k > 0 && k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		zflux[k_off+j_off+i] = 0.5f*(f[k_1_off+j_off+i]+f[k_off+j_off+i])*w[k_off+j_off+i]*art[j_off+i];	
	}
	*/
	
	/*
	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			zflux[k_off+j_off+i] = 0.5f*(f[k_1_off+j_off+i]
											+f[k_off+j_off+i])
									   *w[k_off+j_off+i]*art[j_off+i];	
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			float xflux_tmp, yflux_tmp;
			xflux_tmp = 0.25f*((dt[j_off+i]+dt[j_off+(i-1)])
							  *(f[k_off+j_off+i]
								  +f[k_off+j_off+(i-1)])
							  *u[k_off+j_off+i]);

			yflux_tmp = 0.25f*((dt[j_off+i]+dt[j_1_off+i])
							  *(f[k_off+j_off+i]
								  +f[k_off+j_1_off+i])
							  *v[k_off+j_off+i]);

			xflux_tmp -= 0.5f*(aam[k_off+j_off+i]
											+aam[k_off+j_off+(i-1)])
										*(h[j_off+i]+h[j_off+(i-1)])
										*tprni
										*(fb[k_off+j_off+i]
										 -fb[k_off+j_off+(i-1)])
										*dum[j_off+i]
										/(dx[j_off+i]+dx[j_off+(i-1)]);	


			yflux_tmp -= 0.5f*(aam[k_off+j_off+i]
										    +aam[k_off+j_1_off+i])
										*(h[j_off+i]+h[j_1_off+i])
										*tprni
										*(fb[k_off+j_off+i]
										 -fb[k_off+j_1_off+i])
										*dvm[j_off+i]
										/(dy[j_off+i]+dy[j_1_off+i]);


			xflux[k_off+j_off+i] = 0.5f*(dy[j_off+i]
											+dy[j_off+(i-1)])
									   *xflux_tmp;

			yflux[k_off+j_off+i] = 0.5f*(dx[j_off+i]
											+dx[j_1_off+i])
									   *yflux_tmp;

			zflux[k_off+j_off+i] = 0.5f*(f[k_1_off+j_off+i]
											+f[k_off+j_off+i])
									   *w[k_off+j_off+i]*art[j_off+i];	
		}

		if (var == 'T')
			zflux[j_off+i] = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
		else if (var == 'S')
			zflux[j_off+i] = 0;

		zflux[kb_1_off+j_off+i] = 0;
	}
}


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
				   float dti2, int kb, int jm, int im){

//modify -ff

	//int k = blockDim.z*blockIdx.z + threadIdx.z;
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;
	//float relax;

//! add net horizontal fluxes and then step forward in time

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] += fclim[k][j][i];	
			}
		}
	}
	*/

	/*
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			fb[k_off+j_off+i] += fclim[k_off+j_off+i];	
		}
	}
	*/

	
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				//!730 days, 2000m
				relax = 1.586e-8f*(1.e0f-expf(zz[k]*h[j][i]*5.e-4f));
				//!365 days, 1000m
                //!relax=3.171e-8*(1.e0-exp(zz(k)*h(i,j)*1.e-3)) 
				//!180 days,  500m
                //!relax=6.430e-8*(1.e0-exp(zz(k)*h(i,j)*2.e-3)) 
				//!no relaxation
                //!relax=0.0                                     
				ff[k][j][i] = xflux[k][j][i+1]-xflux[k][j][i]
							 +yflux[k][j+1][i]-yflux[k][j][i]
							 +(zflux[k][j][i]-zflux[k+1][j][i])/dz[k]
							 -relax*fclim[k][j][i]*dt[j][i]*art[j][i];

				ff[k][j][i] = (fb[k][j][i]
								*(h[j][i]+etb[j][i])*art[j][i]
							   -dti2*ff[k][j][i])
							  /((h[j][i]+etf[j][i])*art[j][i]
								*(1.f+relax*dti2));
			}
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			//relax = 1.586e-8f*(1.e0f-expf(zz[k]*h[j_off+i]*5.e-4f));

			//
			//ff[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]
			//				   -xflux[k_off+j_off+i]
			//				   +yflux[k_off+j_A1_off+i]
			//				   -yflux[k_off+j_off+i]
			//				   +(zflux[k_off+j_off+i]
			//						-zflux[k_A1_off+j_off+i])/dz[k]
			//				   -relax*fclim[k_off+j_off+i]
			//					     *dt[j_off+i]*art[j_off+i];

			//ff[k_off+j_off+i] = (fb[k_off+j_off+i]
			//						*(h[j_off+i]
			//							+etb[j_off+i])
			//						*art[j_off+i]
			//					  -dti2*ff[k_off+j_off+i])
			//					/((h[j_off+i]+etf[j_off+i])
			//					 *art[j_off+i]
			//					 *(1.f+relax*dti2));
			//
			//
			//if (relax != relax_aid[k_off+j_off+i]){
			//	printf("relax: %35.25f\n", relax);	
			//	printf("relax_aid: %35.25f\n", relax_aid[k_off+j_off+i]);	
			//	printf("k = %d, j = %d, i = %d\n", k, j, i);
			//}
			//

			ff[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]
							   -xflux[k_off+j_off+i]
							   +yflux[k_off+j_A1_off+i]
							   -yflux[k_off+j_off+i]
							   +(zflux[k_off+j_off+i]
									-zflux[k_A1_off+j_off+i])/dz[k]
							   -relax_aid[k_off+j_off+i]
									*fclim[k_off+j_off+i]
								    *dt[j_off+i]*art[j_off+i];


			ff[k_off+j_off+i] = (fb[k_off+j_off+i]
									*(h[j_off+i]
										+etb[j_off+i])
									*art[j_off+i]
								  -dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])
								 *art[j_off+i]
								 *(1.f+relax_aid[k_off+j_off+i]*dti2));
		}
	}
	*/

	if (blockIdx.x > 0 && blockIdx.x < gridDim.x-1 &&
			blockIdx.y > 0 && blockIdx.y < gridDim.y-1){

		for (k = 0; k < kbm1; k++){
			fb[k_off+j_off+i] += fclim[k_off+j_off+i];	

			ff[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]
							   -xflux[k_off+j_off+i]
							   +yflux[k_off+j_A1_off+i]
							   -yflux[k_off+j_off+i]
							   +(zflux[k_off+j_off+i]
									-zflux[k_A1_off+j_off+i])/dz[k]
							   -relax_aid[k_off+j_off+i]
									*fclim[k_off+j_off+i]
								    *dt[j_off+i]*art[j_off+i];


			ff[k_off+j_off+i] = (fb[k_off+j_off+i]
									*(h[j_off+i]
										+etb[j_off+i])
									*art[j_off+i]
								  -dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])
								 *art[j_off+i]
								 *(1.f+relax_aid[k_off+j_off+i]*dti2));
		}
		fb[kb_1_off+j_off+i] += fclim[kb_1_off+j_off+i];	

	}else{
		if (i < im && j < jm){
			for (k = 0; k < kbm1; k++){
				fb[k_off+j_off+i] += fclim[k_off+j_off+i];	

				if (j > 0 && j < jmm1 && i > 0 && i < imm1){

					ff[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]
									   -xflux[k_off+j_off+i]
									   +yflux[k_off+j_A1_off+i]
									   -yflux[k_off+j_off+i]
									   +(zflux[k_off+j_off+i]
											-zflux[k_A1_off+j_off+i])/dz[k]
									   -relax_aid[k_off+j_off+i]
											*fclim[k_off+j_off+i]
										    *dt[j_off+i]*art[j_off+i];


					ff[k_off+j_off+i] = (fb[k_off+j_off+i]
											*(h[j_off+i]
												+etb[j_off+i])
											*art[j_off+i]
										  -dti2*ff[k_off+j_off+i])
										/((h[j_off+i]+etf[j_off+i])
										 *art[j_off+i]
										 *(1.f+relax_aid[k_off+j_off+i]*dti2));
				}
			}
			fb[kb_1_off+j_off+i] += fclim[kb_1_off+j_off+i];	
		}
	}
}

/*
void advt1_gpu(float fb[][j_size][i_size], float f[][j_size][i_size],
			   float fclim[][j_size][i_size], float ff[][j_size][i_size],
			   float dt[][i_size], float u[][j_size][i_size],
		       float v[][j_size][i_size], float aam[][j_size][i_size],
		       float w[][j_size][i_size], float etb[][i_size],
		       float etf[][i_size]){
*/
/*
void advt1_gpu(float *d_fb, float *d_f,
			   float *d_fclim, float *d_ff,
			   float *d_u, float *d_v, float *d_w, 
		       float *d_etb, float *d_etf, 
		       float *d_aam, float *d_dt){
*/
void advt1_gpu(float *d_fb, float *d_f,
			   float *d_fclim, float *d_ff,
			   char var){

// modify :
//     +reference f fb /only solve the boundary 
//     -          ff
//comments:
//     in this section ,we assign f&fb new value on its boundary(k = kbm1), 
//     but we don't use the boundary values

#ifndef TIME_DISABLE
	struct timeval start_advt1,
				   end_advt1;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advt1);
#endif

	/*
	int i,j,k;
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	float zflux[k_size][j_size][i_size];
	*/

	/*
	dim3 threadPerBlock(block_i, block_j, block_k);
	dim3 blockPerGrid((i_size+block_im1)/block_i, (j_size+block_jm1)/block_j, (k_size+block_km1)/block_k);
	*/

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	float *d_xflux = d_3d_tmp0;
	float *d_yflux = d_3d_tmp1;
	float *d_zflux = d_3d_tmp2;
	/*
	float *d_fb = d_fb_advt1;
	float *d_f = d_f_advt1;
	float *d_fclim = d_fclim_advt1;
	float *d_ff = d_ff_advt1;
	*/

	/*
	checkCudaErrors(cudaMemcpy(d_fb, fb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_f, f, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fclim, fclim, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ff, ff, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	
	advt1_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_f, d_fb, d_fclim,
			kb, jm, im);

	advt1_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_xflux, d_yflux, d_zflux, d_f, d_fb,
			d_u, d_v, d_w, d_aam, d_dt, d_tsurf,
			d_h, d_dx, d_dy, 
			d_dum, d_dvm, d_art, 
			var, tprni, kb, jm, im);
	
	
	//checkCudaErrors(cudaDeviceSynchronize());

//! add net horizontal fluxes and then step forward in time

	advt1_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
			d_xflux, d_yflux, d_zflux, 
			d_ff, d_fb, d_fclim, d_etb, d_etf, d_dt,
			d_relax_aid, d_art, d_h, d_dz, d_zz,
			dti2, kb, jm, im);

	/*
	checkCudaErrors(cudaMemcpy(ff, d_ff, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(f, d_f, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fb, d_fb, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/


#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advt1);
		advt1_time += time_consumed(&start_advt1, 
									&end_advt1);
#endif

	return;
}


__global__ void
smol_adif_kernel_0(float * __restrict__ ff, 
				   float * __restrict__ fsm, 
				   int kb, int jm, int im){

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

#ifdef D3_BLOCK
	if (k < kb && j < jm && i < im){
		ff[k_off+j_off+i] *= fsm[j_off+i];
	}

#else
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			ff[k_off+j_off+i] *= fsm[j_off+i];
		}
	}
#endif
}

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
				   int kb, int jm, int im){

	//modify +ff,xmassflux,ymassflux,zmassflux
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

    float mol;
    float udx,u2dt,vdy,v2dt,wdz,w2dt;
    float value_min=1.e-9,epsilon=1.0e-14;

	/*
	for (k = 0; k < kb-1; k++){
		if (j > 0 && j < jm-1 && i > 0 && i < im){
			if (ff[k_off+j_off+i] < value_min || 
				ff[k_off+j_off+(i-1)] < value_min){
				xmassflux[k_off+j_off+i] = 0;
			}else{
				udx = ABS(xmassflux[k_off+j_off+i]);
				u2dt = dti2*xmassflux[k_off+j_off+i]
						   *xmassflux[k_off+j_off+i]
						   *2.0f/(aru[j_off+i]
									*(dt[j_off+(i-1)]+dt[j_off+i]));
				mol = (ff[k_off+j_off+i]
						-ff[k_off+j_off+(i-1)])
					 /(ff[k_off+j_off+(i-1)]
						+ff[k_off+j_off+i]
						+epsilon);
				xmassflux[k_off+j_off+i] = (udx-u2dt)*mol*sw;
				if (udx < ABS(u2dt))
					xmassflux[k_off+j_off+i] = 0;
			}
		}
	}

	for (k = 0; k < kb-1; k++){
		if (j > 0 && j < jm && i > 0 && i < im-1){
			if (ff[k_off+j_off+i] < value_min || 
				ff[k_off+j_1_off+i] < value_min)
				ymassflux[k_off+j_off+i] = 0;
			else{
				vdy = ABS(ymassflux[k_off+j_off+i]);
				v2dt = dti2*ymassflux[k_off+j_off+i]
						   *ymassflux[k_off+j_off+i]
						   *2.0f
						   /(arv[j_off+i]
								*(dt[j_1_off+i]+dt[j_off+i]));
				mol = (ff[k_off+j_off+i]
						-ff[k_off+j_1_off+i])
					 /(ff[k_off+j_1_off+i]
						+ff[k_off+j_off+i]+epsilon);

				ymassflux[k_off+j_off+i] = (vdy-v2dt)*mol*sw;

				if (vdy < ABS(v2dt))
					ymassflux[k_off+j_off+i] = 0;
			}
		
		}
	}


	for (k = 1; k < kb-1; k++){
		if (j > 0 && j < jm-1 && i > 0 && i < im-1){
			if (ff[k_off+j_off+i] < value_min || 
				ff[k_1_off+j_off+i] < value_min)
				zwflux[k_off+j_off+i] = 0;
			else{
				wdz = ABS(zwflux[k_off+j_off+i]);	
				w2dt = dti2*zwflux[k_off+j_off+i]
						   *zwflux[k_off+j_off+i]
						   /(dzz[k-1]*dt[j_off+i]);
				mol = (ff[k_1_off+j_off+i]
						-ff[k_off+j_off+i])
					 /(ff[k_off+j_off+i]
						+ff[k_1_off+j_off+i]
						+epsilon);
				zwflux[k_off+j_off+i] = (wdz-w2dt)*mol*sw;
				if (wdz < ABS(w2dt))
					zwflux[k_off+j_off+i] = 0;
			}
		}
	}
	*/

	if (j > 0 && j < jm-1 && i > 0 && i < im){
		if (ff[j_off+i] < value_min || 
			ff[j_off+(i-1)] < value_min){
			xmassflux[j_off+i] = 0;
		}else{
			udx = ABS(xmassflux[j_off+i]);
			u2dt = dti2*xmassflux[j_off+i]
					   *xmassflux[j_off+i]
					   *2.0f/(aru[j_off+i]
								*(dt[j_off+(i-1)]+dt[j_off+i]));
			mol = (ff[j_off+i]
					-ff[j_off+(i-1)])
				 /(ff[j_off+(i-1)]
					+ff[j_off+i]
					+epsilon);
			xmassflux[j_off+i] = (udx-u2dt)*mol*sw;
			if (udx < ABS(u2dt))
				xmassflux[j_off+i] = 0;
		}
	}
	

	if (j > 0 && j < jm && i > 0 && i < im-1){
		if (ff[j_off+i] < value_min || 
			ff[j_1_off+i] < value_min)
			ymassflux[j_off+i] = 0;
		else{
			vdy = ABS(ymassflux[j_off+i]);
			v2dt = dti2*ymassflux[j_off+i]
					   *ymassflux[j_off+i]
					   *2.0f
					   /(arv[j_off+i]
							*(dt[j_1_off+i]+dt[j_off+i]));
			mol = (ff[j_off+i]
					-ff[j_1_off+i])
				 /(ff[j_1_off+i]
					+ff[j_off+i]+epsilon);

			ymassflux[j_off+i] = (vdy-v2dt)*mol*sw;

			if (vdy < ABS(v2dt))
				ymassflux[j_off+i] = 0;
		}
	}

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 1; k < kb-1; k++){
			if (j < jm-1){
				if (ff[k_off+j_off+i] < value_min || 
					ff[k_off+j_off+(i-1)] < value_min){
					xmassflux[k_off+j_off+i] = 0;
				}else{
					udx = ABS(xmassflux[k_off+j_off+i]);
					u2dt = dti2*xmassflux[k_off+j_off+i]
							   *xmassflux[k_off+j_off+i]
							   *2.0f/(aru[j_off+i]
										*(dt[j_off+(i-1)]+dt[j_off+i]));
					mol = (ff[k_off+j_off+i]
							-ff[k_off+j_off+(i-1)])
						 /(ff[k_off+j_off+(i-1)]
							+ff[k_off+j_off+i]
							+epsilon);
					xmassflux[k_off+j_off+i] = (udx-u2dt)*mol*sw;
					if (udx < ABS(u2dt))
						xmassflux[k_off+j_off+i] = 0;
				}
			}

			if (i < im-1){
				if (ff[k_off+j_off+i] < value_min || 
					ff[k_off+j_1_off+i] < value_min)
					ymassflux[k_off+j_off+i] = 0;
				else{
					vdy = ABS(ymassflux[k_off+j_off+i]);
					v2dt = dti2*ymassflux[k_off+j_off+i]
							   *ymassflux[k_off+j_off+i]
							   *2.0f
							   /(arv[j_off+i]
									*(dt[j_1_off+i]+dt[j_off+i]));
					mol = (ff[k_off+j_off+i]
							-ff[k_off+j_1_off+i])
						 /(ff[k_off+j_1_off+i]
							+ff[k_off+j_off+i]+epsilon);

					ymassflux[k_off+j_off+i] = (vdy-v2dt)*mol*sw;

					if (vdy < ABS(v2dt))
						ymassflux[k_off+j_off+i] = 0;
				}
			
			}

			if (j < jm-1 && i < im-1){
				if (ff[k_off+j_off+i] < value_min || 
					ff[k_1_off+j_off+i] < value_min)
					zwflux[k_off+j_off+i] = 0;
				else{
					wdz = ABS(zwflux[k_off+j_off+i]);	
					w2dt = dti2*zwflux[k_off+j_off+i]
							   *zwflux[k_off+j_off+i]
							   /(dzz[k-1]*dt[j_off+i]);
					mol = (ff[k_1_off+j_off+i]
							-ff[k_off+j_off+i])
						 /(ff[k_off+j_off+i]
							+ff[k_1_off+j_off+i]
							+epsilon);
					zwflux[k_off+j_off+i] = (wdz-w2dt)*mol*sw;
					if (wdz < ABS(w2dt))
						zwflux[k_off+j_off+i] = 0;
				}
			}
		}
	}
}


//__global__ void
//advt2_gpu_kernel_0(float * __restrict__ xmassflux, 
//				   float * __restrict__ ymassflux, 
//				   float * __restrict__ zwflux, 
//				   const float * __restrict__ u, 
//				   const float * __restrict__ v, 
//				   const float * __restrict__ w,
//				   float * __restrict__ fb, 
//				   float * __restrict__ fbmem, 
//				   const float * __restrict__ dt,
//				   float * __restrict__ eta, 
//				   const float * __restrict__ etb, 
//				   const float * __restrict__ dx, 
//				   const float * __restrict__ dy, 
//				   int kb, int jm, int im){
//
//	//modify -xmassflux -ymassflux -fbmem -eta -zwflux
//	//       +fb
//#ifdef D3_BLOCK
//	int k = blockDim.z*blockIdx.z + threadIdx.z;
//#else
//	int k;
//#endif
//
//	int j = blockDim.y*blockIdx.y + threadIdx.y;
//	int i = blockDim.x*blockIdx.x + threadIdx.x;
//
//	int kbm1 = kb-1;
//	int jmm1 = jm-1;
//	int imm1 = im-1;
//
//#ifdef D3_BLOCK
//	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < im){
//		xmassflux[k_off+j_off+i] = 0.25f*(dy[j_off+(i-1)]+dy[j_off+i])*(dt[j_off+(i-1)]+dt[j_off+i])*u[k_off+j_off+i];
//	}
//
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < imm1){
//		ymassflux[k_off+j_off+i] = 0.25f*(dx[j_1_off+i]+dx[j_off+i])*(dt[j_1_off+i]+dt[j_off+i])*v[k_off+j_off+i];	
//	}
//
//	if (k == 0 && j < jm && i < im){
//		fb[kb_1_off+j_off+i] = fb[kbm1_1_off+j_off+i];
//		eta[j_off+i] = etb[j_off+i];
//	}
//
//	if (k < kb && j < jm && i < im){
//		zwflux[k_off+j_off+i] = w[k_off+j_off+i];
//		fbmem[k_off+j_off+i] = fb[k_off+j_off+i];
//	}
//
//#else
//	/*
//	for (k = 0; k < kb; k++){
//		for (j = 0; j < jm; j++){
//			for (i = 0; i < im; i++){
//				xmassflux[k][j][i] = 0.0;
//				ymassflux[k][j][i] = 0.0;
//			}
//		}
//	}
//	*/
//	for (k = 0; k < kb; k++){
//		if (j < jm && i < im){
//			xmassflux[k_off+j_off+i] = 0;	
//			ymassflux[k_off+j_off+i] = 0;	
//		}
//	}
//
//	for (k = 0; k < kbm1; k++){
//		if (j > 0 && j < jmm1 && i > 0 && i < im){
//			xmassflux[k_off+j_off+i] = 0.25f*(dy[j_off+(i-1)]
//											 +dy[j_off+i])
//											*(dt[j_off+(i-1)]
//											 +dt[j_off+i])
//											*u[k_off+j_off+i];
//		}
//	}
//
//	for (k = 0; k < kbm1; k++){
//		if (j > 0 && j < jm && i > 0 && i < imm1){
//			ymassflux[k_off+j_off+i] = 0.25f*(dx[j_1_off+i]
//											 +dx[j_off+i])
//											*(dt[j_1_off+i]
//											 +dt[j_off+i])
//											*v[k_off+j_off+i];	
//		}
//	}
//
//	if (j < jm && i < im){
//		fb[kb_1_off+j_off+i] = fb[kbm1_1_off+j_off+i];
//		eta[j_off+i] = etb[j_off+i];
//	}
//
//	for (k = 0; k < kb; k++){
//		if (j < jm && i < im){
//			zwflux[k_off+j_off+i] = w[k_off+j_off+i];
//			fbmem[k_off+j_off+i] = fb[k_off+j_off+i];
//		}
//	}
//#endif
//}

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
				   int kb, int jm, int im){

	//modify -xmassflux -ymassflux -fbmem -eta -zwflux
	//       +fb
	int k;

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kb-1; k++){
			xmassflux[k_off+j_off+i] = 0;	
			ymassflux[k_off+j_off+i] = 0;	

			if (j < jmm1){
				xmassflux[k_off+j_off+i] = 0.25f*(dy[j_off+(i-1)]
												 +dy[j_off+i])
												*(dt[j_off+(i-1)]
												 +dt[j_off+i])
												*u[k_off+j_off+i];
			}
			if (i < imm1){
				ymassflux[k_off+j_off+i] = 0.25f*(dx[j_1_off+i]
												 +dx[j_off+i])
												*(dt[j_1_off+i]
												 +dt[j_off+i])
												*v[k_off+j_off+i];	
			}
		}
	}

	if (j < jm && i < im){
		fb[kb_1_off+j_off+i] = fb[kbm1_1_off+j_off+i];
		eta[j_off+i] = etb[j_off+i];
		for (k = 0; k < kb; k++){
			zwflux[k_off+j_off+i] = w[k_off+j_off+i];
			fbmem[k_off+j_off+i] = fb[k_off+j_off+i];
		}
	}
}

/*
advt2_gpu_kernel_1(float *xflux, float *yflux, float *zflux, 
				   float *xmassflux, float *ymassflux, float *zwflux,
				   float *ff, float *f, float *fbmem, 
				   float *w, float *eta, float *etf,
				   float *dz, float *h, float *art, 
				   float dti2, int itera,
				   int kb, int jm, int im){
*/

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
				   int itera, int kb, int jm, int im){

	//modify -xflux,yflux,zflux,

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

#ifdef D3_BLOCK
	float zflux_tmp;
	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
		xflux[k_off+j_off+i] = 0.5f*((xmassflux[k_off+j_off+i]+ABS(xmassflux[k_off+j_off+i]))
									 *fbmem[k_off+j_off+i-1]
									 +(xmassflux[k_off+j_off+i]-ABS(xmassflux[k_off+j_off+i]))
									 *fbmem[k_off+j_off+i]);		

		yflux[k_off+j_off+i] = 0.5f*((ymassflux[k_off+j_off+i]+ABS(ymassflux[k_off+j_off+i]))
									 *fbmem[k_off+j_1_off+i]
									 +(ymassflux[k_off+j_off+i]-ABS(ymassflux[k_off+j_off+i]))
									 *fbmem[k_off+j_off+i]);
	}
	
	if (k == 0 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		zflux[j_off+i] = 0;
		if (itera == 1)
			zflux[j_off+i] = w[j_off+i]*f[j_off+i]*art[j_off+i];
		zflux[kb_1_off+j_off+i] = 0;
	}

	if (k > 0 && k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		zflux_tmp = 0.5f*((zwflux[k_off+j_off+i]+ABS(zwflux[k_off+j_off+i]))
									 *fbmem[k_off+j_off+i] 
									+(zwflux[k_off+j_off+i]-ABS(zwflux[k_off+j_off+i]))
									 *fbmem[k_1_off+j_off+i]);	

		zflux[k_off+j_off+i] = zflux_tmp*art[j_off+i];
	}

#else
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jm && i > 0 && i < im){
			xflux[k_off+j_off+i] = 0.5f*((xmassflux[k_off+j_off+i]
											+ABS(xmassflux[k_off+j_off+i]))
										  *fbmem[k_off+j_off+i-1]
										 +(xmassflux[k_off+j_off+i]
											-ABS(xmassflux[k_off+j_off+i]))
										  *fbmem[k_off+j_off+i]);		

			yflux[k_off+j_off+i] = 0.5f*((ymassflux[k_off+j_off+i]
										    +ABS(ymassflux[k_off+j_off+i]))
										  *fbmem[k_off+j_1_off+i]
										 +(ymassflux[k_off+j_off+i]
											-ABS(ymassflux[k_off+j_off+i]))
										  *fbmem[k_off+j_off+i]);
		}
	}
	
	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		zflux[j_off+i] = 0;
		if (itera == 1){
			if (var == 'T')
				zflux[j_off+i] = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
			if (var == 'S')
				zflux[j_off+i] = 0;
		}
		zflux[kb_1_off+j_off+i] = 0;
	}

	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			zflux[k_off+j_off+i] = 0.5f*((zwflux[k_off+j_off+i]
											+ABS(zwflux[k_off+j_off+i]))
										  *fbmem[k_off+j_off+i] 
										+(zwflux[k_off+j_off+i]
											-ABS(zwflux[k_off+j_off+i]))
										 *fbmem[k_1_off+j_off+i]);	

			zflux[k_off+j_off+i] *= art[j_off+i];
		}
	}
#endif
}

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
				   float dti2, int kb, int jm, int im){

	//modify -ff
#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

#ifdef D3_BLOCK
	float ff_tmp;
	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		ff_tmp = xflux[k_off+j_off+(i+1)]-xflux[k_off+j_off+i]
						   +yflux[k_off+j_A1_off+i]-yflux[k_off+j_off+i]
						   +(zflux[k_off+j_off+i]-zflux[k_A1_off+j_off+i])/dz[k];	
		

		
		ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
								*((double)((h[j_off+i]+eta[j_off+i])*art[j_off+i]))
									-dti2*ff_tmp)
							/((double)((h[j_off+i]+etf[j_off+i])*art[j_off+i]));
		
	}
		
#else
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			ff[k_off+j_off+i] = xflux[k_off+j_off+(i+1)]
							   -xflux[k_off+j_off+i]
							   +yflux[k_off+j_A1_off+i]
							   -yflux[k_off+j_off+i]
							   +(zflux[k_off+j_off+i]
									-zflux[k_A1_off+j_off+i])/dz[k];	
			

			
			ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
									*(h[j_off+i]+eta[j_off+i])
									*art[j_off+i]
								-dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);
			
		}
		
	}
#endif
	
}

__global__ void
advt2_gpu_kernel_12_0(float * __restrict__ ff, 
				   const float * __restrict__ xmassflux, 
				   const float * __restrict__ ymassflux, 
				   const float * __restrict__ zwflux,
				   const float * __restrict__ f, 
				   const float * __restrict__ fbmem, 
				   const float * __restrict__ w, 
				   const float * __restrict__ tsurf, 
				   const float * __restrict__ eta, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   const float * __restrict__ fsm, 
				   const float * __restrict__ dz,
				   float dti2, char var, int itera, 
				   int kb, int jm, int im){
	//modify -xflux,yflux,zflux,

	int k;

	int j = blockDim.y*blockIdx.y + threadIdx.y+1;
	int i = blockDim.x*blockIdx.x + threadIdx.x+1;

	int kbm2 = kb-2;
	int jmm1 = jm-1;
	int imm1 = im-1;

	float xflux, xflux_A1;
	float yflux, yflux_A1;
	float zflux, zflux_A1;

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){

		zflux = 0;
		if (itera == 1){
			if (var == 'T')
				zflux = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
			if (var == 'S')
				zflux = 0;
		}
		
		for (k = 0; k < kbm2; k++){
			xflux = 0.5f*((xmassflux[k_off+j_off+i]
								+ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i-1]
						 +(xmassflux[k_off+j_off+i]
								-ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);		

			xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
								+ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i]
						 +(xmassflux[k_off+j_off+i+1]
								-ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i+1]);		

			yflux = 0.5f*((ymassflux[k_off+j_off+i]
								+ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_1_off+i]
				 		 +(ymassflux[k_off+j_off+i]
								-ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);

			yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
								+ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_off+i]
				 		 +(ymassflux[k_off+j_A1_off+i]
								-ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_A1_off+i]);


			zflux_A1 = 0.5f*((zwflux[k_A1_off+j_off+i]
									+ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_A1_off+j_off+i] 
							+(zwflux[k_A1_off+j_off+i]
									-ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_off+j_off+i]);	

			zflux_A1 *= art[j_off+i];

			ff[k_off+j_off+i] = xflux_A1-xflux
							   +yflux_A1-yflux
							   +(zflux-zflux_A1)/dz[k];	
			

			
			ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
									*(h[j_off+i]+eta[j_off+i])
									*art[j_off+i]
								-dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

			ff[k_off+j_off+i] *= fsm[j_off+i];

			zflux = zflux_A1;
		}

		xflux = 0.5f*((xmassflux[k_off+j_off+i]
							+ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i-1]
					 +(xmassflux[k_off+j_off+i]
							-ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);		

		xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
							+ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i]
					 +(xmassflux[k_off+j_off+i+1]
							-ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i+1]);		

		yflux = 0.5f*((ymassflux[k_off+j_off+i]
							+ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_1_off+i]
			 		 +(ymassflux[k_off+j_off+i]
							-ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);

		yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
							+ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_off+i]
			 		 +(ymassflux[k_off+j_A1_off+i]
							-ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_A1_off+i]);
		zflux_A1 = 0;
		ff[k_off+j_off+i] = xflux_A1-xflux
						   +yflux_A1-yflux
						   +(zflux-zflux_A1)/dz[k];	
		

		
		ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
								*(h[j_off+i]+eta[j_off+i])
								*art[j_off+i]
							-dti2*ff[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

		ff[k_off+j_off+i] *= fsm[j_off+i];
		ff[kb_1_off+j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
advt2_inner_gpu_kernel_12_0(float * __restrict__ ff, 
				   const float * __restrict__ xmassflux, 
				   const float * __restrict__ ymassflux, 
				   const float * __restrict__ zwflux,
				   const float * __restrict__ f, 
				   const float * __restrict__ fbmem, 
				   const float * __restrict__ w, 
				   const float * __restrict__ tsurf, 
				   const float * __restrict__ eta, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   const float * __restrict__ fsm, 
				   const float * __restrict__ dz,
				   float dti2, char var, int itera, 
				   int kb, int jm, int im){
	//modify -xflux,yflux,zflux,

	int k;

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1;

	int kbm2 = kb-2;
	//int jmm1 = jm-1;
	//int imm1 = im-1;

	float xflux, xflux_A1;
	float yflux, yflux_A1;
	float zflux, zflux_A1;

	//if (j > 0 && j < jmm1 && i > 0 && i < imm1){
	if (j < jm-33 && j > 32 && i < im-33 && i > 32){

		zflux = 0;
		if (itera == 1){
			if (var == 'T')
				zflux = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
			if (var == 'S')
				zflux = 0;
		}
		
		for (k = 0; k < kbm2; k++){
			xflux = 0.5f*((xmassflux[k_off+j_off+i]
								+ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i-1]
						 +(xmassflux[k_off+j_off+i]
								-ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);		

			xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
								+ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i]
						 +(xmassflux[k_off+j_off+i+1]
								-ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i+1]);		

			yflux = 0.5f*((ymassflux[k_off+j_off+i]
								+ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_1_off+i]
				 		 +(ymassflux[k_off+j_off+i]
								-ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);

			yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
								+ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_off+i]
				 		 +(ymassflux[k_off+j_A1_off+i]
								-ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_A1_off+i]);


			zflux_A1 = 0.5f*((zwflux[k_A1_off+j_off+i]
									+ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_A1_off+j_off+i] 
							+(zwflux[k_A1_off+j_off+i]
									-ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_off+j_off+i]);	

			zflux_A1 *= art[j_off+i];

			ff[k_off+j_off+i] = xflux_A1-xflux
							   +yflux_A1-yflux
							   +(zflux-zflux_A1)/dz[k];	
			

			
			ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
									*(h[j_off+i]+eta[j_off+i])
									*art[j_off+i]
								-dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

			ff[k_off+j_off+i] *= fsm[j_off+i];

			zflux = zflux_A1;
		}

		xflux = 0.5f*((xmassflux[k_off+j_off+i]
							+ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i-1]
					 +(xmassflux[k_off+j_off+i]
							-ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);		

		xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
							+ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i]
					 +(xmassflux[k_off+j_off+i+1]
							-ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i+1]);		

		yflux = 0.5f*((ymassflux[k_off+j_off+i]
							+ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_1_off+i]
			 		 +(ymassflux[k_off+j_off+i]
							-ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);

		yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
							+ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_off+i]
			 		 +(ymassflux[k_off+j_A1_off+i]
							-ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_A1_off+i]);
		zflux_A1 = 0;
		ff[k_off+j_off+i] = xflux_A1-xflux
						   +yflux_A1-yflux
						   +(zflux-zflux_A1)/dz[k];	
		

		
		ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
								*(h[j_off+i]+eta[j_off+i])
								*art[j_off+i]
							-dti2*ff[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

		ff[k_off+j_off+i] *= fsm[j_off+i];
		ff[kb_1_off+j_off+i] *= fsm[j_off+i];
	}
}


__global__ void
advt2_ew_gpu_kernel_12_0(float * __restrict__ ff, 
				   const float * __restrict__ xmassflux, 
				   const float * __restrict__ ymassflux, 
				   const float * __restrict__ zwflux,
				   const float * __restrict__ f, 
				   const float * __restrict__ fbmem, 
				   const float * __restrict__ w, 
				   const float * __restrict__ tsurf, 
				   const float * __restrict__ eta, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   const float * __restrict__ fsm, 
				   const float * __restrict__ dz,
				   float dti2, char var, int itera, 
				   int kb, int jm, int im){
	//modify -xflux,yflux,zflux,

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;
	}

	int kbm2 = kb-2;
	int jmm1 = jm-1;
	//int imm1 = im-1;

	float xflux, xflux_A1;
	float yflux, yflux_A1;
	float zflux, zflux_A1;

	//if (j > 0 && j < jmm1 && i > 0 && i < imm1){
	if (j < jmm1){

		zflux = 0;
		if (itera == 1){
			if (var == 'T')
				zflux = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
			if (var == 'S')
				zflux = 0;
		}
		
		for (k = 0; k < kbm2; k++){
			xflux = 0.5f*((xmassflux[k_off+j_off+i]
								+ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i-1]
						 +(xmassflux[k_off+j_off+i]
								-ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);		

			xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
								+ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i]
						 +(xmassflux[k_off+j_off+i+1]
								-ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i+1]);		

			yflux = 0.5f*((ymassflux[k_off+j_off+i]
								+ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_1_off+i]
				 		 +(ymassflux[k_off+j_off+i]
								-ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);

			yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
								+ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_off+i]
				 		 +(ymassflux[k_off+j_A1_off+i]
								-ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_A1_off+i]);


			zflux_A1 = 0.5f*((zwflux[k_A1_off+j_off+i]
									+ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_A1_off+j_off+i] 
							+(zwflux[k_A1_off+j_off+i]
									-ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_off+j_off+i]);	

			zflux_A1 *= art[j_off+i];

			ff[k_off+j_off+i] = xflux_A1-xflux
							   +yflux_A1-yflux
							   +(zflux-zflux_A1)/dz[k];	
			

			
			ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
									*(h[j_off+i]+eta[j_off+i])
									*art[j_off+i]
								-dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

			ff[k_off+j_off+i] *= fsm[j_off+i];

			zflux = zflux_A1;
		}

		xflux = 0.5f*((xmassflux[k_off+j_off+i]
							+ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i-1]
					 +(xmassflux[k_off+j_off+i]
							-ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);		

		xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
							+ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i]
					 +(xmassflux[k_off+j_off+i+1]
							-ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i+1]);		

		yflux = 0.5f*((ymassflux[k_off+j_off+i]
							+ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_1_off+i]
			 		 +(ymassflux[k_off+j_off+i]
							-ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);

		yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
							+ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_off+i]
			 		 +(ymassflux[k_off+j_A1_off+i]
							-ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_A1_off+i]);
		zflux_A1 = 0;
		ff[k_off+j_off+i] = xflux_A1-xflux
						   +yflux_A1-yflux
						   +(zflux-zflux_A1)/dz[k];	
		

		
		ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
								*(h[j_off+i]+eta[j_off+i])
								*art[j_off+i]
							-dti2*ff[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

		ff[k_off+j_off+i] *= fsm[j_off+i];
		ff[kb_1_off+j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
advt2_sn_gpu_kernel_12_0(float * __restrict__ ff, 
				   const float * __restrict__ xmassflux, 
				   const float * __restrict__ ymassflux, 
				   const float * __restrict__ zwflux,
				   const float * __restrict__ f, 
				   const float * __restrict__ fbmem, 
				   const float * __restrict__ w, 
				   const float * __restrict__ tsurf, 
				   const float * __restrict__ eta, 
				   const float * __restrict__ etf, 
				   const float * __restrict__ h, 
				   const float * __restrict__ art, 
				   const float * __restrict__ fsm, 
				   const float * __restrict__ dz,
				   float dti2, char var, int itera, 
				   int kb, int jm, int im){
	//modify -xflux,yflux,zflux,

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	int kbm2 = kb-2;
	//int jmm1 = jm-1;
	//int imm1 = im-1;

	float xflux, xflux_A1;
	float yflux, yflux_A1;
	float zflux, zflux_A1;

	//if (j > 0 && j < jmm1 && i > 0 && i < imm1){
	if (i > 32 && i < im-33){

		zflux = 0;
		if (itera == 1){
			if (var == 'T')
				zflux = tsurf[j_off+i]*w[j_off+i]*art[j_off+i];
			if (var == 'S')
				zflux = 0;
		}
		
		for (k = 0; k < kbm2; k++){
			xflux = 0.5f*((xmassflux[k_off+j_off+i]
								+ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i-1]
						 +(xmassflux[k_off+j_off+i]
								-ABS(xmassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);		

			xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
								+ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i]
						 +(xmassflux[k_off+j_off+i+1]
								-ABS(xmassflux[k_off+j_off+i+1]))
							  *fbmem[k_off+j_off+i+1]);		

			yflux = 0.5f*((ymassflux[k_off+j_off+i]
								+ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_1_off+i]
				 		 +(ymassflux[k_off+j_off+i]
								-ABS(ymassflux[k_off+j_off+i]))
							  *fbmem[k_off+j_off+i]);

			yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
								+ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_off+i]
				 		 +(ymassflux[k_off+j_A1_off+i]
								-ABS(ymassflux[k_off+j_A1_off+i]))
							  *fbmem[k_off+j_A1_off+i]);


			zflux_A1 = 0.5f*((zwflux[k_A1_off+j_off+i]
									+ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_A1_off+j_off+i] 
							+(zwflux[k_A1_off+j_off+i]
									-ABS(zwflux[k_A1_off+j_off+i]))
								*fbmem[k_off+j_off+i]);	

			zflux_A1 *= art[j_off+i];

			ff[k_off+j_off+i] = xflux_A1-xflux
							   +yflux_A1-yflux
							   +(zflux-zflux_A1)/dz[k];	
			

			
			ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
									*(h[j_off+i]+eta[j_off+i])
									*art[j_off+i]
								-dti2*ff[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

			ff[k_off+j_off+i] *= fsm[j_off+i];

			zflux = zflux_A1;
		}

		xflux = 0.5f*((xmassflux[k_off+j_off+i]
							+ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i-1]
					 +(xmassflux[k_off+j_off+i]
							-ABS(xmassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);		

		xflux_A1 = 0.5f*((xmassflux[k_off+j_off+i+1]
							+ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i]
					 +(xmassflux[k_off+j_off+i+1]
							-ABS(xmassflux[k_off+j_off+i+1]))
						  *fbmem[k_off+j_off+i+1]);		

		yflux = 0.5f*((ymassflux[k_off+j_off+i]
							+ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_1_off+i]
			 		 +(ymassflux[k_off+j_off+i]
							-ABS(ymassflux[k_off+j_off+i]))
						  *fbmem[k_off+j_off+i]);

		yflux_A1 = 0.5f*((ymassflux[k_off+j_A1_off+i]
							+ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_off+i]
			 		 +(ymassflux[k_off+j_A1_off+i]
							-ABS(ymassflux[k_off+j_A1_off+i]))
						  *fbmem[k_off+j_A1_off+i]);
		zflux_A1 = 0;
		ff[k_off+j_off+i] = xflux_A1-xflux
						   +yflux_A1-yflux
						   +(zflux-zflux_A1)/dz[k];	
		

		
		ff[k_off+j_off+i] = (fbmem[k_off+j_off+i]
								*(h[j_off+i]+eta[j_off+i])
								*art[j_off+i]
							-dti2*ff[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i])*art[j_off+i]);

		ff[k_off+j_off+i] *= fsm[j_off+i];
		ff[kb_1_off+j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
advt2_gpu_kernel_3(float * __restrict__ eta, 
				   const float * __restrict__ etf,
				   const float * __restrict__ ff, 
				   float * __restrict__ fbmem, 
				   int kb, int jm, int im){

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

#ifdef D3_BLOCK
	if (k == 0 && j < jm && i < im){
		eta[j_off+i] = etf[j_off+i];
	}

	if (k < kb && j < jm && i < im){
		fbmem[k_off+j_off+i] = ff[k_off+j_off+i];
	}

#else
	if (j < jm && i < im){
		eta[j_off+i] = etf[j_off+i];
	}

	if (j < jm && i < im){
		for (k = 0; k < kb; k++){
			fbmem[k_off+j_off+i] = ff[k_off+j_off+i];
		}
	}
#endif
}

__global__ void
advt2_gpu_kernel_4(float * __restrict__ fb, 
				   const float * __restrict__ fclim,
				   int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (j < jm && i < im){
		for (k = 0; k < kb; k++){
			fb[k_off+j_off+i] -= fclim[k_off+j_off+i];	
		}
	}

}

//__global__ void
//advt2_gpu_kernel_5(const float * __restrict__ fb, 
//				   const float * __restrict__ ff, 
//				   float *xmassflux, float *ymassflux,
//				   float * __restrict__ xflux, 
//				   float * __restrict__ yflux,
//				   const float * __restrict__ aam, 
//				   const float * __restrict__ etf, 
//				   const float * __restrict__ h, 
//				   const float * __restrict__ art,
//				   const float * __restrict__ dum, 
//				   const float * __restrict__ dvm, 
//				   const float * __restrict__ dx, 
//				   const float * __restrict__ dy, 
//				   float tprni, float dti2,
//				   int kb, int jm, int im){
//	
//#ifdef D3_BLOCK
//	int k = blockDim.z*blockIdx.z + threadIdx.z;
//	float xmassflux, ymassflux;
//#else
//	int k;
//#endif
//
//	int j = blockDim.y*blockIdx.y + threadIdx.y;
//	int i = blockDim.x*blockIdx.x + threadIdx.x;
//
//	int kbm1 = kb-1;
//
//#ifdef D3_BLOCK
//
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
//		xmassflux = 0.5f*(aam[k_off+j_off+i] + aam[k_off+j_off+(i-1)]);	
//		ymassflux = 0.5f*(aam[k_off+j_off+i] + aam[k_off+j_1_off+i]);
//	}
//	
//	if (k < kbm1 && j > 0 && j < jm && i > 0 && i < im){
//		xflux[k_off+j_off+i] = (-xmassflux)*(h[j_off+i]+h[j_off+(i-1)])
//							   *tprni*((fb[k_off+j_off+i]-fclim[k_off+j_off+i])
//									   -(fb[k_off+j_off+(i-1)]-fclim[k_off+j_off+(i-1)]))
//							   *dum[j_off+i]*(dy[j_off+i]+dy[j_off+(i-1)])
//							   *0.5f/(dx[j_off+i]+dx[j_off+(i-1)]);
//
//		yflux[k_off+j_off+i] = (-ymassflux)*(h[j_off+i]+h[j_1_off+i])
//							   *tprni*((fb[k_off+j_off+i]-fclim[k_off+j_off+i])
//									   -(fb[k_off+j_1_off+i]-fclim[k_off+j_1_off+i]))
//							   *dvm[j_off+i]*(dx[j_off+i]+dx[j_1_off+i])
//							   *0.5f/(dy[j_off+i]+dy[j_1_off+i]);
//	}
//#else
//	/*
//	if (k < kb && j < jm && i < im){
//		fb[k*jm*im+j*im+i] -= fclim[k*jm*im+j*im+i];	
//	}
//	*/
//	
//	for (k = 0; k < kbm1; k++){
//		if (j > 0 && j < jm && i > 0 && i < im){
//			xmassflux[k_off+j_off+i] = 0.5f*(aam[k_off+j_off+i]
//											+aam[k_off+j_off+(i-1)]);	
//			ymassflux[k_off+j_off+i] = 0.5f*(aam[k_off+j_off+i]
//											+aam[k_off+j_1_off+i]);
//		}
//	}
//	
//	for (k = 0; k < kbm1; k++){
//		if (j > 0 && j < jm && i > 0 && i < im){
//			xflux[k_off+j_off+i] = (-xmassflux[k_off+j_off+i])
//								   *(h[j_off+i]+h[j_off+(i-1)])
//								   *tprni
//								   *(fb[k_off+j_off+i]
//									 -fb[k_off+j_off+(i-1)])
//								   *dum[j_off+i]
//								   *(dy[j_off+i]+dy[j_off+(i-1)])
//								   *0.5f/(dx[j_off+i]+dx[j_off+(i-1)]);
//
//			yflux[k_off+j_off+i] = (-ymassflux[k_off+j_off+i])
//								   *(h[j_off+i]+h[j_1_off+i])
//								   *tprni
//								   *(fb[k_off+j_off+i]
//									 -fb[k_off+j_1_off+i])
//								   *dvm[j_off+i]
//								   *(dx[j_off+i]+dx[j_1_off+i])
//								   *0.5f/(dy[j_off+i]+dy[j_1_off+i]);
//		}
//	}
//	
//	/*
//	if (k < kb && j < jm && i < im){
//		fb[k*jm*im+j*im+i] += fclim[k*jm*im+j*im+i];	
//	}
//	*/
//#endif
//
//}

__global__ void
advt2_gpu_kernel_5(const float * __restrict__ fb, 
				   const float * __restrict__ ff, 
				   //float *xmassflux, float *ymassflux,
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
				   int kb, int jm, int im){
	
	int k;

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;


	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			float xmassflux = 0.5f*(aam[k_off+j_off+i]
									+aam[k_off+j_off+(i-1)]);	
			float ymassflux = 0.5f*(aam[k_off+j_off+i]
									+aam[k_off+j_1_off+i]);

			xflux[k_off+j_off+i] = (-xmassflux)
								   *(h[j_off+i]+h[j_off+(i-1)])
								   *tprni
								   *(fb[k_off+j_off+i]
									 -fb[k_off+j_off+(i-1)])
								   *dum[j_off+i]
								   *(dy[j_off+i]+dy[j_off+(i-1)])
								   *0.5f/(dx[j_off+i]+dx[j_off+(i-1)]);

			yflux[k_off+j_off+i] = (-ymassflux)
								   *(h[j_off+i]+h[j_1_off+i])
								   *tprni
								   *(fb[k_off+j_off+i]
									 -fb[k_off+j_1_off+i])
								   *dvm[j_off+i]
								   *(dx[j_off+i]+dx[j_1_off+i])
								   *0.5f/(dy[j_off+i]+dy[j_1_off+i]);
		}
	}
}

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
				   int kb, int jm, int im){

#ifdef D3_BLOCK
	int k = blockDim.z*blockIdx.z + threadIdx.z;
#else
	int k;
#endif

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

#ifdef D3_BLOCK

	if (k < kbm1 && j > 0 && j < jmm1 && i > 0 && i < imm1){
		ff[k_off+j_off+i] = ff[k_off+j_off+i] 
							-dti2*(xflux[k_off+j_off+(i+1)]-xflux[k_off+j_off+i]
									+yflux[k_off+j_A1_off+i]-yflux[k_off+j_off+i])
								/(double)((h[j_off+i]+etf[j_off+i])*art[j_off+i]);	
	}
#else

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] += fclim[k][j][i];	
			}
		}
	}
	*/
	if (j < jm && i < im){
		for (k = 0; k < kb; k++){
			fb[k_off+j_off+i] += fclim[k_off+j_off+i];	
		}
	}

	/*
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			for (k = 0; k < kbm1; k++){
				ff[k][j][i] = ff[k][j][i]
							 -dti2*(xflux[k][j][i+1]
								   -xflux[k][j][i]
								   +yflux[k][j+1][i]
								   -yflux[k][j][i])
								/((h[j][i]+etf[j][i])*art[j][i]);	
			}
		}
	}
	*/

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 0; k < kbm1; k++){
			ff[k_off+j_off+i] = ff[k_off+j_off+i] 
								-dti2*(xflux[k_off+j_off+(i+1)]
									  -xflux[k_off+j_off+i]
									  +yflux[k_off+j_A1_off+i]
									  -yflux[k_off+j_off+i])
									/((h[j_off+i]+etf[j_off+i])
										*art[j_off+i]);	
		}
	}

#endif
}

/*
void advt2_gpu(float fb[][j_size][i_size],   float f[][j_size][i_size],
			   float fclim[][j_size][i_size],float ff[][j_size][i_size],
		       float etb[][i_size],          float u[][j_size][i_size],
		       float v[][j_size][i_size],    float etf[][i_size], 
		       float aam[][j_size][i_size],  float w[][j_size][i_size],
		       float dt[][i_size]){
*/

/*
void advt2_gpu(float *d_fb,   float *d_f,
			   float *d_fclim,float *d_ff,
		       float *d_u, float *d_v, float *d_w,  
		       float *d_etb, float *d_etf, 
		       float *d_aam, float *d_dt){
*/

void advt2_gpu(float *d_fb,   float *d_f,
			   float *d_fclim,float *d_ff,
			   float *d_ff_east, float *d_ff_west,
			   float *d_ff_south, float *d_ff_north,
			   char var){

#ifndef TIME_DISABLE
	struct timeval start_advt2,
				   end_advt2;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advt2);
#endif


	//int i, j, k;
	int itera;	
	
	/*
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	float zflux[k_size][j_size][i_size];
	

	float fbmem[k_size][j_size][i_size];
	float eta[j_size][i_size];

	float xmassflux[k_size][j_size][i_size];
	float ymassflux[k_size][j_size][i_size];
	float zwflux[k_size][j_size][i_size];
	*/
	

	float *d_xflux = d_3d_tmp0;
	float *d_yflux = d_3d_tmp1;
	//float *d_zflux = d_3d_tmp2;

	float *d_fbmem = d_3d_tmp3;
	float *d_eta = d_2d_tmp0;

	float *d_xmassflux = d_3d_tmp4;
	float *d_ymassflux = d_3d_tmp5;
	float *d_zwflux = d_3d_tmp6;

#ifdef D3_BLOCK
	dim3 threadPerBlock(block_i_3D, block_j_3D, block_k_3D);
	dim3 blockPerGrid((im+block_i_3D-1)/block_i_3D, (jm+block_j_3D-1)/block_j_3D, (kb+block_k_3D-1)/block_k_3D);

	dim3 threadPerBlock_3x2(block_i_3D, block_k_3D, block_j_3D);
	dim3 blockPerGrid_3x2((im+block_i_3D-1)/block_i_3D, (jm+block_k_3D-1)/block_k_3D, (kb+block_j_3D-1)/block_j_3D);
#else

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 threadPerBlock_3x2(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);
	dim3 blockPerGrid_3x2((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);
#endif

	//printf("In advt2 iint = %d", iint);
	
	
	/*
	checkCudaErrors(cudaMemcpy(d_fb, fb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_f, f, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fclim, fclim, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ff, ff, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam, aam, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/

	//modify -xmassflux -ymassflux -fbmem -eta -zwflux
	//       +fb
	/*
	advt2_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xmassflux_advt2, d_ymassflux_advt2, d_zwflux_advt2,
			d_u, d_v, d_w, d_fb, d_fbmem_advt2, d_dt, d_eta_advt2, d_etb,
			d_dx, d_dy, kb, jm, im);
	*/
	advt2_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xmassflux, d_ymassflux, d_zwflux,
			d_u, d_v, d_w, d_fb, d_fbmem, d_dt, d_eta, d_etb,
			d_dx, d_dy, kb, jm, im);

	//checkCudaErrors(cudaDeviceSynchronize());

	for (itera = 1; itera <= nitera; itera++){
		////modify -xflux,yflux,zflux,
		//advt2_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
		//		d_xflux, d_yflux, d_zflux,
		//		d_xmassflux, d_ymassflux, d_zwflux,
		//		d_f, d_fbmem, d_w, d_tsurf, d_art,
		//		var,
		//		itera, kb, jm, im);

		////modify -ff
		//advt2_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
		//		d_xflux, d_yflux, d_zflux, 
		//		d_ff, d_fbmem, d_eta, d_etf,
		//		d_h, d_art, d_dz, dti2, kb, jm, im);

		//advt2_gpu_kernel_12_0<<<blockPerGrid, threadPerBlock>>>(
		//		d_ff, d_xmassflux, d_ymassflux, d_zwflux,
		//		d_f, d_fbmem, d_w, d_tsurf, d_eta, d_etf,
		//		d_h, d_art, d_fsm, d_dz, 
		//		dti2, var, itera, kb, jm, im);
		
		//exchange3d_mpi_gpu(d_ff, im, jm, kbm1);
		//exchange3d_cuda_aware_mpi(d_ff, im, jm, kbm1);

		advt2_ew_gpu_kernel_12_0<<<blockPerGrid_ew_32, 
								   threadPerBlock_ew_32,
								   0, stream[1]>>>(
				d_ff, d_xmassflux, d_ymassflux, d_zwflux,
				d_f, d_fbmem, d_w, d_tsurf, d_eta, d_etf,
				d_h, d_art, d_fsm, d_dz, 
				dti2, var, itera, kb, jm, im);

		advt2_sn_gpu_kernel_12_0<<<blockPerGrid_sn_32, 
								   threadPerBlock_sn_32,
								   0, stream[2]>>>(
				d_ff, d_xmassflux, d_ymassflux, d_zwflux,
				d_f, d_fbmem, d_w, d_tsurf, d_eta, d_etf,
				d_h, d_art, d_fsm, d_dz, 
				dti2, var, itera, kb, jm, im);

		advt2_inner_gpu_kernel_12_0<<<blockPerGrid, 
								   threadPerBlock,
								   0, stream[0]>>>(
				d_ff, d_xmassflux, d_ymassflux, d_zwflux,
				d_f, d_fbmem, d_w, d_tsurf, d_eta, d_etf,
				d_h, d_art, d_fsm, d_dz, 
				dti2, var, itera, kb, jm, im);
		
		checkCudaErrors(cudaStreamSynchronize(stream[1]));
		checkCudaErrors(cudaStreamSynchronize(stream[2]));

		exchange3d_cudaUVA(d_ff, 
						   d_ff_east, d_ff_west, 
						   d_ff_south, d_ff_north, 
						   stream[1], im,jm,kbm1);

		checkCudaErrors(cudaStreamSynchronize(stream[0]));


		//modify +ff,xmassflux,ymassflux,zmassflux

		//smol_adif_kernel_0<<<blockPerGrid, threadPerBlock>>>(
		//		d_ff, d_fsm, kb, jm, im);

		smol_adif_kernel_1<<<blockPerGrid, threadPerBlock>>>(
				d_ff, d_dt, d_xmassflux, d_ymassflux, d_zwflux, 
				d_aru, d_arv, d_dzz, sw, dti2, kb, jm, im);

		//checkCudaErrors(cudaDeviceSynchronize());

		//modify -eta, fbmem
		//advt2_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
		//		d_eta_advt2, d_etf, d_ff, d_fbmem_advt2, kb, jm, im);

		advt2_gpu_kernel_3<<<blockPerGrid, threadPerBlock>>>(
				d_eta, d_etf, d_ff, d_fbmem, kb, jm, im);
		
		//checkCudaErrors(cudaDeviceSynchronize());
	}
	
	advt2_gpu_kernel_4<<<blockPerGrid, threadPerBlock>>>(
			d_fb, d_fclim, 
			kb, jm, im);

	advt2_gpu_kernel_5<<<blockPerGrid, threadPerBlock>>>(
			d_fb, d_ff, //d_xmassflux, d_ymassflux,
			d_xflux, d_yflux, d_aam, d_etf, 
			d_h, d_art, d_dum, d_dvm, d_dx, d_dy, 
			tprni, dti2, kb, jm, im);
	

	//checkCudaErrors(cudaDeviceSynchronize());
	
	advt2_gpu_kernel_6<<<blockPerGrid, threadPerBlock>>>(
			d_ff, d_fb, d_fclim, d_etf, d_xflux, d_yflux, 
			d_h, d_art, 
			dti2, kb, jm, im);
	
	/*
	checkCudaErrors(cudaMemcpy(ff, d_ff, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fb, d_fb, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	*/
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advt2);
		advt2_time += time_consumed(&start_advt2, 
									&end_advt2);
#endif

	return;
}



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
				   int kb, int jm, int im){

	//		+ f
	int k, ki;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};

	float dh;
	float a[k_size], c[k_size];
	float ee[k_size], gg[k_size];
	float rad[k_size];


//#ifdef optimize
//
//	float dh;
//	double a[k_size], c[k_size], ee[k_size], gg[k_size];
//	float rad[k_size];
//
//	if (j < jm && i < im){
//		dh = h[j_off+i]+etf[j_off+i];	
//	}
//
//	for (k = 1; k < kbm1; k++){
//		if (j < jm && i < im){
//			
//			a[k-1] = -dti2*(kh[k_off+j_off+i]+umol)
//									/(dz[k-1]*dzz[k-1]
//									 *dh*dh);
//
//			
//			c[k] = -dti2*(kh[k_off+j_off+i]+umol) /
//									(dz[k]*dzz[k-1]*dh*dh);
//			
//		}
//	}
//
//	for (k = 0; k < kb; k++){
//		//if (j < jm && i < im){
//			rad[k] = 0;	
//		//}
//	}
//
//	if (nbc == 2 || nbc == 4){
//		for (k = 0; k < kbm1; k++){ 
//			if (j < jm && i < im){
//				rad[k] = swrad[j_off+i]
//					    *(r[ntp-1]
//							*expf(z[k]*dh/ad1[ntp-1])
//						 +(1.0f-r[ntp-1])
//							*expf(z[k]*dh/ad2[ntp-1]));
//			}
//		}
//	}
//
//	if (nbc == 1){
//		if (j < jm && i < im){
//			ee[0] = a[0]/(a[0]-1.0);
//			
//			gg[0] = dti2*wfsurf[j_off+i]/(dz[0]*dh)-f[j_off+i];
//			
//			gg[0] = gg[0]/(a[0]-1.0);
//			
//		}
//	}else if (nbc == 2){
//		if (j < jm && i < im){
//			ee[0] = a[0]/(a[0]-1.0);
//
//			gg[0] = dti2*(wfsurf[j_off+i]+rad[0]-rad[1])
//						/(dz[0]*dh) 
//				   -f[j_off+i];
//
//			gg[0] = gg[0]/(a[0]-1.0);
//		}
//	}else if (nbc == 3 || nbc == 4){
//		if (j < jm && i < im){
//			ee[0] = 0;
//			gg[0] = fsurf[j_off+i];
//		}
//	}
//
//	if (nbc == 2 || nbc == 4){
//		for (k = 1; k < kbm2; k++){
//			if (j < jm && i < im){
//				double gg_tmp;
//				gg_tmp = 1.0/(a[k]+c[k]*(1.0-ee[k-1])-1.0);
//				ee[k] = a[k]*gg_tmp;
//				gg[k] = (c[k]*gg[k-1]-f[k_off+j_off+i]
//						+dti2*(rad[k]-rad[k+1])/(dh*dz[k]))*gg_tmp;
//			}
//		}
//	}else{
//		
//		for (k = 1; k < kbm2; k++){
//			if (j < jm && i < im){
//				double gg_tmp;
//				gg_tmp = 1.0/(a[k]+c[k]*(1.0-ee[k-1])-1.0);
//				ee[k] = a[k]*gg_tmp;
//				gg[k] = (c[k]*gg[k-1]-f[k_off+j_off+i])*gg_tmp;
//			}
//		}
//		
//	}
//	
//	if (nbc == 2 || nbc == 4){
//		if (j < jm && i < im){
//			f[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
//									-f[kbm1_1_off+j_off+i]
//									+dti2*(rad[kbm1-1]-rad[kb-1])
//										 /(dh*dz[kbm1-1]))
//								   /(c[kbm1-1]
//									*(1.0f-ee[kbm2-1])-1.0);
//		}
//	}else{
//		if (j < jm && i < im){
//			f[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
//									-f[kbm1_1_off+j_off+i])
//									/(c[kbm1-1]
//									*(1.0f-ee[kbm2-1])-1.0);
//		}
//	}
//
//	for (ki = kb-3; ki >= 0; ki--){
//		if (j < jm && i < im){
//			f[ki_off+j_off+i] = (ee[ki]*f[ki_A1_off+j_off+i]
//								+gg[ki]);	
//		}
//	}
//#endif


	/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = h[j][i]+etf[j][i];
		}
	}
	*/

	/*
	if (j < jm && i < im){
		dh = h[j_off+i]+etf[j_off+i];	
	}
	*/



	/*
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k-1][j][i] = -dti2*(kh[k][j][i]+umol)
					/(dz[k-1]*dzz[k-1]*dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(kh[k][j][i]+umol)
					/(dz[k]*dzz[k-1]*dh[j][i]*dh[j][i]);

			}
		}
	}	
	*/

	/*
	for (k = 1; k < kbm1; k++){
		if (j < jm && i < im){
			
			a[k-1] = -dti2*(kh[k_off+j_off+i]+umol)
						  /(dz[k-1]*dzz[k-1]*dh*dh);
			
			c[k] = -dti2*(kh[k_off+j_off+i]+umol)
					    /(dz[k]*dzz[k-1]*dh*dh);
			
		}
	}
	*/
	
	

//! calculate penetrative radiation. At the bottom any unattenuated
//! radiation is deposited in the bottom layer

	/*
	for(k = 0; k < kb; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				rad[k][j][i] = 0.0f;
			}
		}
	}
	*/

	/*
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			rad[k] = 0;	
		}
	}
	*/

	/*
	if(nbc == 2 || nbc == 4 ){
		for(k = 0; k < kbm1; k++){
			for(j = 0; j < jm; j++){
				for(i = 0; i < im; i++){
					rad[k][j][i] = swrad[j][i]*(r[ntp-1]*expf(z[k]*dh[j][i]/ad1[ntp-1])
							+(1.0f-r[ntp-1])*expf(z[k]*dh[j][i]/ad2[ntp-1]));
				}
			}
		}
	}
	*/
	
	/*
	if (nbc == 2 || nbc == 4){
		for (k = 0; k < kbm1; k++){ 
			if (j < jm && i < im){
				rad[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}
	}
	*/
	

	/*
	if(nbc == 1){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0);
				gg[0][j][i] = dti2*wfsurf[j][i]/(dz[0]*dh[j][i])-f[0][j][i];
				gg[0][j][i] = gg[0][j][i]/(a[0][j][i]-1.0);
			}
		}
	}
	*/

	/*
	if (nbc == 1){
		if (j < jm && i < im){
			ee[0] = a[0]/(a[0]-1.0f);
			
			gg[0] = -dti2*wfsurf[j_off+i]/(-dz[0]*dh)
						  -f[j_off+i];
			
			gg[0] = gg[0]/(a[0]-1.0f);
			
		}
	}
	*/
	

	/*
	else if(nbc == 2){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0);
				gg[0][j][i] = dti2*(wfsurf[j][i]+rad[0][j][i]-rad[1][j][i])
								/(dz[0]*dh[j][i])
							  -f[0][j][i];
				gg[0][j][i] = gg[0][j][i]/(a[0][j][i]-1.0);
			}
		}
	}
	*/
	
	/*
	else if (nbc == 2){
		if (j < jm && i < im){
			ee[0] = a[0]/(a[0]-1.0f);

			gg[0] = dti2*(wfsurf[j_off+i]+rad[0]-rad[1])
				        /(dz[0]*dh) 
				    -f[j_off+i];

			gg[0] = gg[0]/(a[0]-1.0f);
		}
	}
	*/
	


	/*
	else if(nbc == 3 || nbc == 4){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				ee[0][j][i]=0.0;
				gg[0][j][i]= fsurf[j][i];
			}
		}
	}
	*/

	/*
	else if (nbc == 3 || nbc == 4){
		if (j < jm && i < im){
			ee[0] = 0;
			gg[0] = fsurf[j_off+i];
		}
	}
	*/
	
	

	/*
	for(k = 1; k < kbm2; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0/(a[k][j][i]+c[k][j][i]*(1.0-ee[k-1][j][i])-1.0);
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (c[k][j][i]*gg[k-1][j][i]-f[k][j][i]
								+dti2*(rad[k][j][i]-rad[k+1][j][i])
									/(dh[j][i]*dz[k]))*gg[k][j][i];
			}
		}
	}
	*/
	
	/*
	if (j < jm && i < im){
		for (k = 1; k < kbm2; k++){
			gg[k] = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])-1.0f);

			ee[k] = a[k]*gg[k];

			gg[k] = (c[k]*gg[k-1]-f[k_off+j_off+i]
					  +dti2*(rad[k]-rad[k+1])/(dh*dz[k]))*gg[k];
		}
	}
	*/
	

	/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			f[kbm1-1][j][i] = (c[kbm1-1][j][i]*gg[kbm2-1][j][i]-f[kbm1-1][j][i]
					+dti2*(rad[kbm1-1][j][i]-rad[kb-1][j][i])
					/(dh[j][i]*dz[kbm1-1]))
				/(c[kbm1-1][j][i]*(1.0f-ee[kbm2-1][j][i])-1.0);	
		}
	}
	*/

	/*
	if (j < jm && i < im){
		f[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
								  -f[kbm1_1_off+j_off+i]
								  +dti2*(rad[kbm1-1]-rad[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(c[kbm1-1]*(1.0f-ee[kbm2-1])-1.0f);
	}
	*/
	

	/*
	for(ki = kb-3; ki >= 0; ki--){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				f[ki][j][i] = (ee[ki][j][i]*f[ki+1][j][i]+gg[ki][j][i]);
			}
		}
	}
	*/
	
	
	/*
	for (ki = kb-3; ki >= 0; ki--){
		if (j < jm && i < im){
			f[ki_off+j_off+i] = (ee[ki]*f[ki_A1_off+j_off+i]+gg[ki]);	
		}
	}
	*/

	
	if (j < jm && i < im){
		dh = h[j_off+i]+etf[j_off+i];	

		for (k = 1; k < kbm1; k++){
			
			a[k-1] = -dti2*(kh[k_off+j_off+i]+umol)
						  /(dz[k-1]*dzz[k-1]*dh*dh);
			
			c[k] = -dti2*(kh[k_off+j_off+i]+umol)
					    /(dz[k]*dzz[k-1]*dh*dh);
			
		}

		for (k = 0; k < kb; k++){
			rad[k] = 0;	
		}
	
		if (nbc == 2 || nbc == 4){
			for (k = 0; k < kbm1; k++){ 
				rad[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		if (nbc == 1){
			ee[0] = a[0]/(a[0]-1.0f);
			
			gg[0] = -dti2*wfsurf[j_off+i]/(-dz[0]*dh)
						  -f[j_off+i];
			
			gg[0] = gg[0]/(a[0]-1.0f);
			
		}else if (nbc == 2){
			ee[0] = a[0]/(a[0]-1.0f);

			gg[0] = dti2*(wfsurf[j_off+i]+rad[0]-rad[1])
				        /(dz[0]*dh) 
				    -f[j_off+i];

			gg[0] = gg[0]/(a[0]-1.0f);

		}else if (nbc == 3 || nbc == 4){
			ee[0] = 0;
			gg[0] = fsurf[j_off+i];
		}

		for (k = 1; k < kbm2; k++){
			gg[k] = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])-1.0f);

			ee[k] = a[k]*gg[k];

			gg[k] = (c[k]*gg[k-1]-f[k_off+j_off+i]
					  +dti2*(rad[k]-rad[k+1])/(dh*dz[k]))*gg[k];
		}

		f[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
								  -f[kbm1_1_off+j_off+i]
								  +dti2*(rad[kbm1-1]-rad[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(c[kbm1-1]*(1.0f-ee[kbm2-1])-1.0f);

		for (ki = kb-3; ki >= 0; ki--){
			f[ki_off+j_off+i] = (ee[ki]*f[ki_A1_off+j_off+i]+gg[ki]);	
		}
	}
}

/*
void proft_gpu(float f[][j_size][i_size], float wfsurf[][i_size],
			   float fsurf[][i_size], int nbc,
		       float etf[][i_size], float kh[][j_size][i_size],
			   float swrad[][i_size]){
*/

/*
void proft_gpu(float *d_f, float *d_wfsurf,
			   float *d_fsurf, 
		       float *d_etf, float *d_kh,
			   float *d_swrad, int nbc){
*/

void proft_gpu(float *d_f, float *d_wfsurf,
			   float *d_fsurf, int nbc){
	//modify:
	//		+ f

#ifndef TIME_DISABLE
	struct timeval start_proft,
				   end_proft;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_proft);
#endif

	/*
	int nbc = *c_nbc;
	int i,j,k,ki;
    double a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    double ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
	float dh[j_size][i_size],rad[k_size][j_size][i_size];


	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};
	*/

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	//float *d_a = d_3d_tmp0;
	//float *d_c = d_3d_tmp1;
	//float *d_ee = d_3d_tmp2;
	//float *d_gg = d_3d_tmp3;

	//float *d_dh = d_2d_tmp0;
	//float *d_rad = d_3d_tmp4;

	
	//it is not right, for wfsurf & fsurf are also formal parameters
	/*
	checkCudaErrors(cudaMemcpy(d_kh, kh, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_swrad, swrad, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wfsurf_proft, wfsurf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_f_proft, f, kb*jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fsurf_proft, fsurf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	*/
	

	/*
	proft_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_kh, d_etf, d_swrad, d_wfsurf_proft, d_f_proft, d_fsurf_proft,
			d_a_proft, d_c_proft, d_ee_proft, d_gg_proft, d_dh_proft, d_rad_proft,
			d_h, d_dz, d_dzz, d_z, dti2, umol, ntp, nbc, kb, jm, im);
	*/
	/*
	proft_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_kh, d_etf, d_swrad, d_wfsurf_proft, 
			d_f_proft, d_fsurf_proft,
			d_a, d_c, d_ee, d_gg, d_dh, d_rad,
			d_h, d_dz, d_dzz, d_z, 
			dti2, umol, ntp, nbc, kb, jm, im);
	*/
	proft_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_kh, d_etf, d_swrad, d_wfsurf, 
			d_f, d_fsurf,
			//d_a, d_c, d_ee, d_gg, d_dh, d_rad,
			d_h, d_dz, d_dzz, d_z, 
			dti2, umol, ntp, nbc, kb, jm, im);

	/*
	checkCudaErrors(cudaMemcpy(f, d_f_proft, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	*/
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_proft);
		proft_time += time_consumed(&start_proft, 
									&end_proft);
#endif

    return;
}



__global__ void
dens_gpu_kernel_0(const float * __restrict__ ti, 
				  const float * __restrict__ si, 
				  float * __restrict__ rhoo,
				  const float * __restrict__ zz, 
				  const float * __restrict__ h, 
				  const float * __restrict__ fsm,
				  float tbias, float sbias,
				  float grav, float rhoref,
				  int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

    float cr,p,rhor,sr,tr,tr2,tr3,tr4;

	int kbm1 = kb-1;

	/*
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				
				tr = ti[k][j][i] + tbias;
				sr = si[k][j][i] + sbias;
				tr2 = tr*tr;
				tr3 = tr2*tr;
				tr4 = tr3*tr;
//! approximate pressure in units of bars
				p = grav*rhoref*(-zz[k]*h[j][i])*1.0e-5f;

				
				rhor=-0.157406e0f+6.793952e-2f*tr
			           -9.095290e-3f*tr2+1.001685e-4f*tr3
			           -1.120083e-6f*tr4+6.536332e-9f*tr4*tr;
				
				rhor=rhor+(0.824493e0f-4.0899e-3f*tr
		                +7.6438e-5f*tr2-8.2467e-7f*tr3
						+5.3875e-9f*tr4)*sr
						+(-5.72466e-3f+1.0227e-4f*tr
						-1.6546e-6f*tr2)*powf(ABS(sr), 1.5f)
						+4.8314e-4f*sr*sr;

				cr=1449.1e0f+0.0821e0f*p+4.55e0f*tr-0.045e0f*tr2
                      +1.34e0f*(sr-35.0e0f);

				rhor=rhor+1.0e5f*p/(cr*cr)*(1.0e0f-2.e0f*p/(cr*cr));

				rhoo[k][j][i]=rhor/rhoref*fsm[j][i];
				
			}
		}
	}
	*/

	
	if (j < jm && i < im){
		for (k = 0; k < kbm1; k++){
			tr = ti[k_off+j_off+i]+tbias;		
			sr = si[k_off+j_off+i]+sbias;
			tr2 = tr*tr;
			tr3 = tr2*tr;
			tr4 = tr3*tr;

			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-5f;


			rhor=-0.157406e0f+6.793952e-2f*tr
			       -9.095290e-3f*tr2+1.001685e-4f*tr3
			       -1.120083e-6f*tr4+6.536332e-9f*tr4*tr;

			//comment by xsz: powf(x, 1.5f) has been converted to x*(sqrt(x))
			rhor=rhor+(0.824493e0f-4.0899e-3f*tr
		            +7.6438e-5f*tr2-8.2467e-7f*tr3
					+5.3875e-9f*tr4)*sr
					+(-5.72466e-3f+1.0227e-4f*tr
					-1.6546e-6f*tr2)*(ABS(sr)*sqrtf(ABS(sr)))
					+4.8314e-4f*sr*sr;


			cr=1449.1e0f+0.0821e0f*p+4.55e0f*tr-0.045e0f*tr2
                  +1.34e0f*(sr-35.0e0f);

			rhor=rhor+1.0e5f*p/(cr*cr)*(1.0e0f-2.e0f*p/(cr*cr));
			
			rhoo[k_off+j_off+i]=rhor/rhoref*fsm[j_off+i];
		}
	}
}

/*
void dens_gpu(float si[][j_size][i_size], 
			  float ti[][j_size][i_size],
			  float rhoo[][j_size][i_size]){
*/
void dens_gpu(float *d_si, float *d_ti, float *d_rhoo){

	//modify -rhoo

#ifndef TIME_DISABLE
	struct timeval start_dens,
				   end_dens;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_dens);
#endif

	//int i,j,k;
    //float cr,p,rhor,sr,tr,tr2,tr3,tr4;

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	/*
	float *d_ti = d_ti_dens;
	float *d_si = d_si_dens;
	float *d_rhoo = d_rhoo_dens;
	*/
	
	/*
	checkCudaErrors(cudaMemcpy(d_ti_dens, ti, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_si_dens, si, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/

	dens_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_ti, d_si, d_rhoo, d_zz, d_h, d_fsm,
		    tbias, sbias, grav, rhoref, kb, jm, im);

	/*
	checkCudaErrors(cudaMemcpy(rhoo, d_rhoo_dens, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_dens);
		dens_time += time_consumed(&start_dens, 
								   &end_dens);
#endif
	
	return;
}



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
				  int kb, int jm, int im){

	int k;// = blockDim.z*blockIdx.z + threadIdx.z;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	//int kbm1 = kb-1;
	int kbm2 = kb-2;

	int jmm1 = jm-1;
	int imm1 = im-1;

//! combine horizontal and vertical advection with coriolis, surface
//! slope and baroclinic terms
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				uf[k][j][i]=advx[k][j][i]
					         +(uf[k][j][i]-uf[k+1][j][i])*aru[j][i]/dz[k]
							 -aru[j][i]*0.25f
								*(cor[j][i]*dt[j][i]*(v[k][j+1][i]+v[k][j][i])
									  +cor[j][i-1]*dt[j][i-1]*(v[k][j+1][i-1]+v[k][j][i-1]))
							 +grav*0.125f*(dt[j][i]+dt[j][i-1])
								*(egf[j][i]-egf[j][i-1]+egb[j][i]-egb[j][i-1]
										+(e_atmos[j][i]-e_atmos[j][i-1])*2.0f)
								*(dy[j][i]+dy[j][i-1])
							 +drhox[k][j][i];
			}
		}
	}
	*/
//!  step forward in time
	/*
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				uf[k][j][i]=((h[j][i]+etb[j][i]+h[j][i-1]+etb[j][i-1])
								*aru[j][i]*ub[k][j][i]
								-2.0f*dti2*uf[k][j][i])
							/((h[j][i]+etf[j][i]+h[j][i-1]+etf[j][i-1])*aru[j][i]);
			}	
		}
	}
	*/

	//for (k = 0; k < kb; k++){
	//	if (j < jm && i < im){
	//		uf[k_off+j_off+i] = 0;	
	//	}
	//}

	//for (k = 1; k < kbm1; k++){
	//	//if (j < jm && i > 0 && i < im){
	//	if (j < jmm1 && j > 0 && i > 0 && i < imm1){
	//		uf[k_off+j_off+i] = 0.25f*(w[k_off+j_off+i]
	//									+w[k_off+j_off+(i-1)])
	//								 *(u[k_off+j_off+i]
	//									+u[k_1_off+j_off+i]);	
	//	}
	//}

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		uf[j_off+i] = 0;	
		uf[kb_1_off+j_off+i] = 0;	

		for (k = 0; k < kbm2; k++){
			uf[k_A1_off+j_off+i] = 0.25f*(w[k_A1_off+j_off+i]
										+w[k_A1_off+j_off+(i-1)])
									 *(u[k_A1_off+j_off+i]
										+u[k_off+j_off+i]);	

			uf[k_off+j_off+i] = advx[k_off+j_off+i]
							   +(uf[k_off+j_off+i]
									-uf[k_A1_off+j_off+i])
								 *aru[j_off+i]/dz[k]
							   -aru[j_off+i]*0.25f
								 *(cor[j_off+i]*dt[j_off+i]
									*(v[k_off+j_A1_off+i]
										+v[k_off+j_off+i])
								  +cor[j_off+(i-1)]*dt[j_off+(i-1)]
									*(v[k_off+j_A1_off+(i-1)]
										+v[k_off+j_off+(i-1)]))
							   +grav*0.125f
								 *(dt[j_off+i]+dt[j_off+(i-1)])
								 *(egf[j_off+i]-egf[j_off+(i-1)]
									+egb[j_off+i]-egb[j_off+(i-1)]
									+(e_atmos[j_off+i]
										-e_atmos[j_off+(i-1)])*2.0f)
								 *(dy[j_off+i]+dy[j_off+(i-1)])
							   +drhox[k_off+j_off+i];
			
	
			uf[k_off+j_off+i] = ((h[j_off+i]
									+etb[j_off+i]
									+h[j_off+(i-1)]
									+etb[j_off+(i-1)])
								  *aru[j_off+i]*ub[k_off+j_off+i]
								 -2.0f*dti2*uf[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i]
									+h[j_off+(i-1)]+etf[j_off+(i-1)])
								  *aru[j_off+i]);
	
		}

		uf[k_off+j_off+i] = advx[k_off+j_off+i]
						   +(uf[k_off+j_off+i]
								-uf[k_A1_off+j_off+i])
							 *aru[j_off+i]/dz[k]
						   -aru[j_off+i]*0.25f
							 *(cor[j_off+i]*dt[j_off+i]
								*(v[k_off+j_A1_off+i]
									+v[k_off+j_off+i])
							  +cor[j_off+(i-1)]*dt[j_off+(i-1)]
								*(v[k_off+j_A1_off+(i-1)]
									+v[k_off+j_off+(i-1)]))
						   +grav*0.125f
							 *(dt[j_off+i]+dt[j_off+(i-1)])
							 *(egf[j_off+i]-egf[j_off+(i-1)]
								+egb[j_off+i]-egb[j_off+(i-1)]
								+(e_atmos[j_off+i]
									-e_atmos[j_off+(i-1)])*2.0f)
							 *(dy[j_off+i]+dy[j_off+(i-1)])
						   +drhox[k_off+j_off+i];
		
	
		uf[k_off+j_off+i] = ((h[j_off+i]
								+etb[j_off+i]
								+h[j_off+(i-1)]
								+etb[j_off+(i-1)])
							  *aru[j_off+i]*ub[k_off+j_off+i]
							 -2.0f*dti2*uf[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i]
								+h[j_off+(i-1)]+etf[j_off+(i-1)])
							  *aru[j_off+i]);
	}

	return;

}

/*
void advu_gpu(float uf[][j_size][i_size], float w[][j_size][i_size],
			  float u[][j_size][i_size], float advx[][j_size][i_size],
			  float dt[][i_size], float v[][j_size][i_size],
			  float egf[][i_size], float egb[][i_size],
			  float e_atmos[][i_size], float drhox[][j_size][i_size],
			  float etb[][i_size], float ub[][j_size][i_size],
			  float etf[][i_size]){
*/

/*
void advu_gpu(float *d_ub, float *d_u, float *d_uf, 
			  float *d_v, float *d_w, 
			  float *d_egb, float *d_egf, 
			  float *d_etb, float *d_etf, 
			  float *d_advx, float *d_drhox,
			  float *d_e_atmos, float *d_dt){
*/
void advu_gpu(){

	//modify: - uf
	//comments: in GPU version ,we ignore the value assigned for uf on j=0,
	//          I believe these values will be set later by MPI communication 
	//comments: above is wrong, after a test in 2013/07/14/, if we don't set 
	//			proper values on the boundary, the result will not concide with
	//			the C/Fortran version
	//comment: ?? I forgot about above 2013/07/25

#ifndef TIME_DISABLE
	struct timeval start_advu,
				   end_advu;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advu);
#endif

	//int i,j,k;

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	/*
	checkCudaErrors(cudaMemcpy(d_ub, ub, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advx, advx, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_egf, egf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_egb, egb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_e_atmos, e_atmos, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drhox, drhox, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	

//! do vertical advection
//! combine horizontal and vertical advection with coriolis, surface
//! slope and baroclinic terms
//!  step forward in time

	advu_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_uf, d_ub, d_u, d_v, d_w, d_advx, 
			d_egf, d_egb, d_etf, d_etb,
			d_dt, d_e_atmos, d_drhox, 
			d_h, d_cor, d_aru, d_dy, d_dz,
			grav, dti2, kb, jm, im);

	
	/*
	checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advu);
		advu_time += time_consumed(&start_advu, 
								   &end_advu);
#endif
	
	return;
}


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
				  int kb, int jm, int im){

	//modify -vf;
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int kbm2 = kb-2;

	int jmm1 = jm-1;
	int imm1 = im-1;
	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				vf[k][j][i] = 0.0f;
			}
		}
	}
	*/

	/*
	for (k = 0; k < kb; k++){
		if (j < jm && i < im){
			vf[k_off+j_off+i] = 0;	
		}
	}
	*/

	/*
	for (k = 1; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 0; i < im; i++){
				vf[k][j][i] = 0.25f*(w[k][j][i]+w[k][j-1][i])*(v[k][j][i]+v[k-1][j][i]);
			}
		}
	}
	*/

	/*
	for (k = 1; k < kbm1; k++){
		if (j > 0 && j < jm && i < im){
			vf[k_off+j_off+i] = 0.25f*(w[k_off+j_off+i]
										+w[k_off+j_1_off+i])
									 *(v[k_off+j_off+i]
										+v[k_1_off+j_off+i]);	
		}
	}
	*/

	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j <jmm1; j++){
			for(i = 1; i <imm1; i++){
				vf[k][j][i] = advy[k][j][i]
								+(vf[k][j][i]-vf[k+1][j][i])*arv[j][i]/dz[k]
								+arv[j][i]*0.25f
									*(cor[j][i]*dt[j][i]*(u[k][j][i+1]+u[k][j][i])
										+cor[j-1][i]*dt[j-1][i]*(u[k][j-1][i+1]+u[k][j-1][i]))
								+grav*0.125f*(dt[j][i]+dt[j-1][i])
									*(egf[j][i]-egf[j-1][i]+egb[j][i]-egb[j-1][i]
										+(e_atmos[j][i]-e_atmos[j-1][i])*2.0f)
									*(dx[j][i]+dx[j-1][i])
								+drhoy[k][j][i];
			}
		}
	}
	*/
	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				vf[k][j][i] = ((h[j][i]+etb[j][i]+h[j-1][i]+etb[j-1][i])
									*arv[j][i]*vb[k][j][i]
							-2.0f*dti2*vf[k][j][i])
								/((h[j][i]+etf[j][i]+h[j-1][i]+etf[j-1][i])
									*arv[j][i]);
			}
		}
	}
	*/

	/*
	for (k = 0; k < kbm1; k++){
		if (j > 0 && j < jmm1 && i > 0 && i < imm1){
			vf[k_off+j_off+i] = advy[k_off+j_off+i]
							   +(vf[k_off+j_off+i]
									-vf[k_A1_off+j_off+i])
								  *arv[j_off+i]/dz[k]
							   +arv[j_off+i]*0.25f
								  *(cor[j_off+i]*dt[j_off+i]
									 *(u[k_off+j_off+(i+1)]
										 +u[k_off+j_off+i])
								   +cor[j_1_off+i]*dt[j_1_off+i]
									 *(u[k_off+j_1_off+(i+1)]
										 +u[k_off+j_1_off+i]))
							   +grav*0.125f*(dt[j_off+i]+dt[j_1_off+i])
								  *(egf[j_off+i]-egf[j_1_off+i]
									 +egb[j_off+i]-egb[j_1_off+i]
									 +(e_atmos[j_off+i]
										 -e_atmos[j_1_off+i])*2.0f)
								  *(dx[j_off+i]+dx[j_1_off+i])
							   +drhoy[k_off+j_off+i];

			vf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i]
										+h[j_1_off+i]+etb[j_1_off+i])
									*arv[j_off+i]*vb[k_off+j_off+i]
								  -2.0f*dti2*vf[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i]
									+h[j_1_off+i]+etf[j_1_off+i])
								  *arv[j_off+i]);
		}
	}
	*/

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		vf[j_off+i] = 0;	
		vf[kb_1_off+j_off+i] = 0;	

		for (k = 0; k < kbm2; k++){

			vf[k_A1_off+j_off+i] = 0.25f*(w[k_A1_off+j_off+i]
										+w[k_A1_off+j_1_off+i])
									 *(v[k_A1_off+j_off+i]
										+v[k_off+j_off+i]);	

			vf[k_off+j_off+i] = advy[k_off+j_off+i]
							   +(vf[k_off+j_off+i]
									-vf[k_A1_off+j_off+i])
								  *arv[j_off+i]/dz[k]
							   +arv[j_off+i]*0.25f
								  *(cor[j_off+i]*dt[j_off+i]
									 *(u[k_off+j_off+(i+1)]
										 +u[k_off+j_off+i])
								   +cor[j_1_off+i]*dt[j_1_off+i]
									 *(u[k_off+j_1_off+(i+1)]
										 +u[k_off+j_1_off+i]))
							   +grav*0.125f*(dt[j_off+i]+dt[j_1_off+i])
								  *(egf[j_off+i]-egf[j_1_off+i]
									 +egb[j_off+i]-egb[j_1_off+i]
									 +(e_atmos[j_off+i]
										 -e_atmos[j_1_off+i])*2.0f)
								  *(dx[j_off+i]+dx[j_1_off+i])
							   +drhoy[k_off+j_off+i];

			vf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i]
										+h[j_1_off+i]+etb[j_1_off+i])
									*arv[j_off+i]*vb[k_off+j_off+i]
								  -2.0f*dti2*vf[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i]
									+h[j_1_off+i]+etf[j_1_off+i])
								  *arv[j_off+i]);
		}

		vf[k_off+j_off+i] = advy[k_off+j_off+i]
						   +(vf[k_off+j_off+i]
								-vf[k_A1_off+j_off+i])
							  *arv[j_off+i]/dz[k]
						   +arv[j_off+i]*0.25f
							  *(cor[j_off+i]*dt[j_off+i]
								 *(u[k_off+j_off+(i+1)]
									 +u[k_off+j_off+i])
							   +cor[j_1_off+i]*dt[j_1_off+i]
								 *(u[k_off+j_1_off+(i+1)]
									 +u[k_off+j_1_off+i]))
						   +grav*0.125f*(dt[j_off+i]+dt[j_1_off+i])
							  *(egf[j_off+i]-egf[j_1_off+i]
								 +egb[j_off+i]-egb[j_1_off+i]
								 +(e_atmos[j_off+i]
									 -e_atmos[j_1_off+i])*2.0f)
							  *(dx[j_off+i]+dx[j_1_off+i])
						   +drhoy[k_off+j_off+i];

		vf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i]
									+h[j_1_off+i]+etb[j_1_off+i])
								*arv[j_off+i]*vb[k_off+j_off+i]
							  -2.0f*dti2*vf[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i]
								+h[j_1_off+i]+etf[j_1_off+i])
							  *arv[j_off+i]);
	}
}

/*
void advv_gpu(float vf[][j_size][i_size], float w[][j_size][i_size],
			  float u[][j_size][i_size], float advy[][j_size][i_size],
			  float dt[][i_size], float v[][j_size][i_size],
			  float egf[][i_size], float egb[][i_size],
			  float e_atmos[][i_size], float drhoy[][j_size][i_size],
			  float etb[][i_size], float vb[][j_size][i_size],
			  float etf[][i_size]){
*/

/*
void advv_gpu(float *d_vb, float *d_v, float *d_vf, 
			  float *d_u, float *d_w,
			  float *d_egb, float *d_egf, 
			  float *d_etb, float *d_etf,
			  float *d_advy, float *d_drhoy,
			  float *d_e_atmos, float *d_dt){
*/
void advv_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_advv,
				   end_advv;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advv);
#endif

	//int i,j,k;
	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);
	
	/*
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vb, vb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w, w, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_egb, egb, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_egf, egf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advy, advy, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_e_atmos, e_atmos, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drhoy, drhoy, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	

//! do vertical advection
//! combine horizontal and vertical advection with coriolis, surface
//! slope and baroclinic terms
		
//! step forward in time
	advv_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_vf, d_v, d_vb, d_u, d_w, 
			d_etb, d_etf, d_egf, d_egb,
			d_advy, d_dt, d_e_atmos, d_drhoy, 
			d_cor, d_arv, d_dx, d_dz, d_h,
			grav, dti2, kb, jm, im);

	
	/*
	checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advv);
		advv_time += time_consumed(&start_advv, 
								   &end_advv);
#endif

    return;
}


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
				   int kb, int jm, int im){

	int k, ki;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	int jmm1 = jm-1;
	int imm1 = im-1;

	float dh, tps;
	//float a[k_size], c[k_size];
	float ee[k_size], gg[k_size];

//#ifdef optimize
//
//	float dh = 1.0f;
//	float tps;
//	float a[k_size], c[k_size], ee[k_size], gg[k_size];
//
//	if (j < jm && j > 0 && i < im && i > 0){
//		dh = (h[j_off+i]+etf[j_off+i]+h[j_off+(i-1)]+etf[j_off+(i-1)])*0.5f;
//	}
//
//	for (k = 0; k < kb; k++){
//		if (j < jm && j > 0 && i < im && i > 0){
//			c[k] = (km[k_off+j_off+i]+km[k_off+j_off+(i-1)])*0.5;
//		}
//	}
//
//	for (k = 1; k < kbm1; k++){
//		if (j < jm && i < im){
//			a[k-1] = -dti2*(c[k]+umol)/(dz[k-1]*dzz[k-1]*dh*dh);
//			c[k] = -dti2*(c[k]+umol)/(dz[k]*dzz[k-1]*dh*dh);
//		} 
//	}
//
//	if (j < jm && i < im){
//		ee[0] = a[0]/(a[0]-1.0);
//		gg[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh)
//						-uf[j_off+i])/(a[0]-1.0);
//	}
//
//
//	for (k = 1; k < kbm2; k++){
//		if (j < jm && i < im){
//			float gg_tmp = 1.0/(a[k]+c[k]*(1.0-ee[k-1])-1.0);
//			ee[k] = a[k]*gg_tmp;
//			gg[k] = (c[k]*gg[k-1]-uf[k_off+j_off+i])*gg_tmp;
//		}
//	}
//
//	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
//		float tmp = 0.25f*(vb[kbm1_1_off+j_off+i]+vb[kbm1_1_off+j_A1_off+i]+vb[kbm1_1_off+j_off+(i-1)]+vb[kbm1_1_off+j_A1_off+(i-1)]);
//		tps = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
//					*sqrtf((ub[kbm1_1_off+j_off+i]*ub[kbm1_1_off+j_off+i])+(tmp*tmp));
//
//		uf[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]-uf[kbm1_1_off+j_off+i])
//								/(tps*dti2/(-dz[kbm1-1]*dh)-1.0f
//								-(ee[kbm2-1]-1.0f)*c[kbm1-1]);
//
//		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];
//	}
//
//	for (ki = kb-3; ki >= 0; ki--){
//		if (j < jmm1 && j > 0 && i < imm1 && i > 0){
//			uf[ki_off+j_off+i] = (ee[ki]*uf[ki_A1_off+j_off+i]+gg[ki])*dum[j_off+i];
//		}
//	}
//
//	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
//		wubot[j_off+i] = -tps*uf[kbm1_1_off+j_off+i];
//	}
//
//#endif

/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = 1.0f;
		}
	}
*/
	/*
	if (j < jm && i < im){
		dh = 1.0f;
	}
	*/
/*
	for(j = 1; j < jm; j++){
		for(i = 1; i < im; i++){
			dh[j][i] = (h[j][i]+etf[j][i]+h[j][i-1]+etf[j][i-1])*0.5f;
		}
	}
*/
	/*
	if (j < jm && j > 0 && i < im && i > 0){
		dh = (h[j_off+i]+etf[j_off+i]
			 +h[j_off+(i-1)]+etf[j_off+(i-1)])*0.5f;
	}
	*/
/*
	for(k = 0; k < kb; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				c[k][j][i] = (km[k][j][i]+km[k][j][i-1])*0.5;
			}
		}
	}
*/
	/*
	for (k = 0; k < kb; k++){
		if (j < jm && j > 0 && i < im && i > 0){
			c[k] = (km[k_off+j_off+i]
				   +km[k_off+j_off+(i-1)])*0.5f;
		}
	}
	*/
/*
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k-1][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k-1]*dzz[k-1]*dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k]*dzz[k-1]*dh[j][i]*dh[j][i]);
			}
		}
	}
*/
	/*
	for (k = 1; k < kbm1; k++){
		if (j < jm && i < im){
			a[k-1] = -dti2*(c[k]+umol)
						 /(dz[k-1]*dzz[k-1]*dh*dh);

			c[k] = -dti2*(c[k]+umol)
						/(dz[k]*dzz[k-1]*dh*dh);
		} 
	}
	*/
/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0);
			gg[0][j][i] = (-dti2*wusurf[j][i]/(-dz[0]*dh[j][i])
							-uf[0][j][i])/(a[0][j][i]-1.0);
		}
	}
*/
	/*
	if (j < jm && i < im){
		ee[0] = a[0]/(a[0]-1.0f);
		gg[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh)
				 -uf[j_off+i])
			   /(a[0]-1.0f);
	}
	*/

/*
	for(k = 1; k < kbm2; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0/(a[k][j][i]+c[k][j][i]*(1.0-ee[k-1][j][i])-1.0);
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (c[k][j][i]*gg[k-1][j][i]-uf[k][j][i])*gg[k][j][i];
			}
		}
	}
*/
	/*
	for (k = 1; k < kbm2; k++){
		if (j < jm && i < im){
			gg[k] = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])-1.0f);

			ee[k] = a[k]*gg[k];

			gg[k] = (c[k]*gg[k-1]-uf[k_off+j_off+i])*gg[k];
		}
	}
	*/
/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			float tmp = 0.25f*(vb[kbm1-1][j][i]+vb[kbm1-1][j+1][i]+vb[kbm1-1][j][i-1]+vb[kbm1-1][j+1][i-1]);
				tps[j][i] = 0.5f*(cbc[j][i]+cbc[j][i-1])
						 *sqrtf((ub[kbm1-1][j][i]*ub[kbm1-1][j][i])+(tmp*tmp));

			uf[kbm1-1][j][i] = (c[kbm1-1][j][i]*gg[kbm2-1][j][i]-uf[kbm1-1][j][i])
								/(tps[j][i]*dti2/(-dz[kbm1-1]*dh[j][i])-1.0f
						          -(ee[kbm2-1][j][i]-1.0f)*c[kbm1-1][j][i]);
			uf[kbm1-1][j][i] = uf[kbm1-1][j][i]*dum[j][i];
		}
	}
*/
	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		float tmp = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tmp*tmp));

		uf[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps*dti2/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c[kbm1-1]);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];
	}
	*/
/*
	for(ki = kb-3; ki >= 0; ki--){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				uf[ki][j][i] = (ee[ki][j][i]*uf[ki+1][j][i]+gg[ki][j][i])*dum[j][i];
			}
		}
	}
*/
	/*
	for (ki = kb-3; ki >= 0; ki--){
		if (j < jmm1 && j > 0 && i < imm1 && i > 0){
			uf[ki_off+j_off+i] = (ee[ki]*uf[ki_A1_off+j_off+i]
								 +gg[ki])*dum[j_off+i];
		}
	}
	*/
/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			wubot[j][i] = -tps[j][i]*uf[kbm1-1][j][i];
		}
	}
*/
	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		wubot[j_off+i] = -tps*uf[kbm1_1_off+j_off+i];
	}
	*/


	if (j < jmm1 && j > 0 && i < imm1 && i > 0){

		dh = (h[j_off+i]+etf[j_off+i]
			 +h[j_off+(i-1)]+etf[j_off+(i-1)])*0.5f;

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]+km[jm*im+j_off+(i-1)])*0.5f;
		a_tmp = -dti2*(c_tmp+umol)/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh)
				 -uf[j_off+i])
			   /(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
					/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_off+(i-1)])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
						 /(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-uf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
						/(dz[k+1]*dzz[k]*dh*dh);
		}

		//c_tmp_next = (km[kbm1_1_off+j_off+i]
		//			 +km[kbm1_1_off+j_off+(i-1)])*0.5f;

		//c_tmp = -dti2*(c_tmp_next+umol)
		//			/(dz[kbm2]*dzz[kbm2-1]*dh*dh);

		float tps_tmp = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp*tps_tmp));

		uf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps*dti2/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee[ki]*uf[ki_A1_off+j_off+i]
								 +gg[ki])*dum[j_off+i];
		}

		wubot[j_off+i] = -tps*uf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profu_inner_gpu_kernel_0(
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
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y;
	const int i = blockDim.x*blockIdx.x + threadIdx.x;

	const int kbm1 = kb-1;
	const int kbm2 = kbm1-1;

	float dh, tps;
	float ee[k_size], gg[k_size];


	if (j < jm-33 && j > 32 && i < im-33 && i > 32){
	//if (j < jm-1 && j > 0 && i < im-33 && i > 32){

		dh = (h[j_off+i]+etf[j_off+i]
			 +h[j_off+(i-1)]+etf[j_off+(i-1)])*0.5f;

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]+km[jm*im+j_off+(i-1)])*0.5f;
		a_tmp = -dti2*(c_tmp+umol)/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh)
				 -uf[j_off+i])
			   /(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
					/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_off+(i-1)])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
						 /(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-uf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
						/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tps_tmp = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp*tps_tmp));

		uf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps*dti2/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee[ki]*uf[ki_A1_off+j_off+i]
								 +gg[ki])*dum[j_off+i];
		}

		wubot[j_off+i] = -tps*uf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profu_ew_gpu_kernel_0(
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
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;
	}

	const int kbm1 = kb-1;
	const int kbm2 = kbm1-1;

	float dh, tps;
	float ee[k_size], gg[k_size];


	if (j < jm-1){

		dh = (h[j_off+i]+etf[j_off+i]
			 +h[j_off+(i-1)]+etf[j_off+(i-1)])*0.5f;

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]+km[jm*im+j_off+(i-1)])*0.5f;
		a_tmp = -dti2*(c_tmp+umol)/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh)
				 -uf[j_off+i])
			   /(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
					/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_off+(i-1)])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
						 /(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-uf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
						/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tps_tmp = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp*tps_tmp));

		uf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps*dti2/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee[ki]*uf[ki_A1_off+j_off+i]
								 +gg[ki])*dum[j_off+i];
		}

		wubot[j_off+i] = -tps*uf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profu_sn_gpu_kernel_0(
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
				   int kb, int jm, int im){

	int k, ki;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	const int kbm1 = kb-1;
	const int kbm2 = kbm1-1;

	float dh, tps;
	float ee[k_size], gg[k_size];


	if (i > 32 && i < im-33){

		dh = (h[j_off+i]+etf[j_off+i]
			 +h[j_off+(i-1)]+etf[j_off+(i-1)])*0.5f;

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]+km[jm*im+j_off+(i-1)])*0.5f;
		a_tmp = -dti2*(c_tmp+umol)/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh)
				 -uf[j_off+i])
			   /(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
					/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_off+(i-1)])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
						 /(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-uf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
						/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tps_tmp = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp*tps_tmp));

		uf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps*dti2/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee[ki]*uf[ki_A1_off+j_off+i]
								 +gg[ki])*dum[j_off+i];
		}

		wubot[j_off+i] = -tps*uf[kbm1_1_off+j_off+i];
	}
}

/*
void profu_gpu(float etf[][i_size],float km[][j_size][i_size],
			   float wusurf[][i_size],float uf[][j_size][i_size],
			   float vb[][j_size][i_size],float ub[][j_size][i_size],
			   float wubot[][i_size]){
*/

/*
void profu_gpu(float *d_ub, float *d_uf, 
			   float *d_wusurf, float *d_wubot,
			   float *d_vb, float *d_km,
			   float *d_etf){
*/
void profu_gpu(){

//modify +uf -wubot

#ifndef TIME_DISABLE
	struct timeval start_profu,
				   end_profu;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_profu);
#endif

	/*
	int i,j,k,ki;
    double a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    double ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
	float dh[j_size][i_size];
	float tps[j_size][i_size];
	*/

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	dim3 threadPerBlock_ew(32, 4);
	dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	dim3 threadPerBlock_sn(32, 4);
	dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	//float *d_a = d_3d_tmp0;
	//float *d_c = d_3d_tmp1;
	//float *d_ee = d_3d_tmp2;
	//float *d_gg = d_3d_tmp3;

	//float *d_dh = d_2d_tmp0;
	//float *d_tps = d_2d_tmp1;
	
	/*
	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wusurf, wusurf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vb, vb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ub, ub, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/
	

	/*
	profu_gpu_kernel<<<blockPerGrid, threadPerBlock>>>(
			d_dh_profu, d_tps_profu, 
			d_a_profu, d_c_profu, d_ee_profu, d_gg_profu, 
		    d_uf, d_ub, d_vb,  d_km, d_etf, d_wusurf, d_wubot, 
			d_cbc, d_dum, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);
	*/

	//profu_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		//d_dh, d_tps, d_a, d_c, d_ee, d_gg, 
	//	    d_uf, d_ub, d_vb,  d_km, d_etf, d_wusurf, d_wubot, 
	//		d_cbc, d_dum, d_h, d_dz, d_dzz, 
	//		dti2, umol, kb, jm, im);

/*
#ifdef CUDA_SLICE_MPI
    exchange2d_mpi_gpu(d_wubot,im,jm);
#else
	checkCudaErrors(cudaMemcpy(wubot, d_wubot, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	
    //exchange2d_mpi_xsz_(wubot,im,jm);
    exchange2d_mpi(wubot,im,jm);

	checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
#endif
*/
    //exchange2d_mpi_gpu(d_wubot,im,jm);
    //exchange2d_cuda_aware_mpi(d_wubot,im,jm);
	
	/*
	checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/

	profu_ew_gpu_kernel_0<<<blockPerGrid_ew, threadPerBlock_ew,
						 0, stream[1]>>>(
		    d_uf, d_ub, d_vb,  d_km, d_etf, d_wusurf, d_wubot, 
			d_cbc, d_dum, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);
	profu_sn_gpu_kernel_0<<<blockPerGrid_sn, threadPerBlock_sn,
						 0, stream[2]>>>(
		    d_uf, d_ub, d_vb,  d_km, d_etf, d_wusurf, d_wubot, 
			d_cbc, d_dum, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);

	profu_inner_gpu_kernel_0<<<blockPerGrid, threadPerBlock,
						 0, stream[0]>>>(
		    d_uf, d_ub, d_vb,  d_km, d_etf, d_wusurf, d_wubot, 
			d_cbc, d_dum, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

    //exchange2d_mpi_gpu(d_wubot, im,jm);

    exchange2d_cudaUVA(d_wubot, 
					   d_wubot_east, d_wubot_west, 
					   d_wubot_south, d_wubot_north, 
					   stream[1], im, jm);

	//MPI_Barrier(pom_comm);
    //exchange2d_cuda_ipc(d_wubot, d_wubot_east, d_wubot_west, 
	//					stream[1], im, jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[0]));
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_profu);
		profu_time += time_consumed(&start_profu, 
									&end_profu);
#endif

    return;
}



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
				   int kb, int jm, int im){

	int k, ki;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

	float dh, tps;
	//float a[k_size], c[k_size];
	float ee[k_size], gg[k_size];

//#ifdef optimize
//	float dh = 1.0f;
//	float tps;
//	float a[k_size], c[k_size], ee[k_size], gg[k_size];
//
//	if (j < jm && j > 0 && i < im && i > 0){
//		dh = 0.5f*(h[j_off+i]+etf[j_off+i]+h[j_1_off+i]+etf[j_1_off+i]);
//	}
//
//	for (k = 0; k < kb; k++){
//		if (j < jm && j > 0 && i < im && i > 0){
//			c[k] = (km[k_off+j_off+i]+km[k_off+j_1_off+i])*0.5f;
//		}
//	}
//
//	for (k = 1; k < kbm1; k++){
//		if (j < jm && i < im){
//			a[k-1] = -dti2*(c[k]+umol)/(dz[k-1]*dzz[k-1]*dh*dh);
//			c[k] = -dti2*(c[k]+umol)/(dz[k]*dzz[k-1]*dh*dh);
//		}
//	}
//
//	if (j < jm && i < im){
//		ee[0] = a[0]/(a[0]-1.0f);
//		gg[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh)-vf[j_off+i])/(a[0]-1.0f);
//	}
//
//	for (k = 1; k < kbm2; k++){
//		if (j < jm && i < im){
//			float gg_tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])-1.0f);
//			ee[k] = a[k]*gg_tmp;
//			gg[k] = (c[k]*gg[k-1]-vf[k_off+j_off+i])*gg_tmp;
//		}
//	}
//
//	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
//		float tmp = 0.25f*(ub[kbm1_1_off+j_off+i]+ub[kbm1_1_off+j_off+i+1]+ub[kbm1_1_off+j_1_off+i]+ub[kbm1_1_off+j_1_off+(i+1)]);
//		tps = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
//				  *sqrtf((tmp*tmp)+(vb[kbm1_1_off+j_off+i]*vb[kbm1_1_off+j_off+i]));
//		vf[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]-vf[kbm1_1_off+j_off+i])
//								/(tps*dti2/(-dz[kbm1-1]*dh)-1.0f
//										-(ee[kbm2-1]-1.0f)*c[kbm1-1]);
//		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];
//	}
//
//	for (ki = kb-3; ki >=0; ki--){
//		if (j < jmm1 && j > 0 && i < imm1 && i > 0){
//			vf[ki_off+j_off+i] = (ee[ki]*vf[ki_A1_off+j_off+i]+gg[ki])*dvm[j_off+i];
//		}
//	}
//
//	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
//		wvbot[j_off+i] = -tps*vf[kbm1_1_off+j_off+i];
//	}
//#endif

/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = 1.0f;
		}
	}
*/
	/*
	if (j < jm && i < im){
		dh = 1.0f;
	}
	*/
/*
	for(j = 1; j < jm; j++){
		for(i = 1; i < im; i++){
			dh[j][i] = 0.5f*(h[j][i]+etf[j][i]+h[j-1][i]+etf[j-1][i]);
		}
	}
*/
	/*
	if (j < jm && j > 0 && i < im && i > 0){
		dh = 0.5f*(h[j_off+i]+etf[j_off+i]
				  +h[j_1_off+i]+etf[j_1_off+i]);
	}
	*/
/*
	for(k = 0; k < kb; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				c[k][j][i] = (km[k][j][i]+km[k][j-1][i])*0.5f;
			}
		}
	}
*/
	/*
	for (k = 0; k < kb; k++){
		if (j < jm && j > 0 && i < im && i > 0){
			c[k] = (km[k_off+j_off+i]
				   +km[k_off+j_1_off+i])*0.5f;
		}
	}
	*/
/*
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k-1][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k-1]*dzz[k-1]*dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k]*dzz[k-1]*dh[j][i]*dh[j][i]);
			}
		}
	}
*/
	/*
	for (k = 1; k < kbm1; k++){
		if (j < jm && i < im){
			a[k-1] = -dti2*(c[k]+umol)
				 	/(dz[k-1]*dzz[k-1]*dh*dh);

			c[k] = -dti2*(c[k]+umol)
				 	/(dz[k]*dzz[k-1]*dh*dh);
		}
	}
	*/
/*
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0f);
			gg[0][j][i] = (-dti2*wvsurf[j][i]/(-dz[0]*dh[j][i])-vf[0][j][i])
				/(a[0][j][i]-1.0f);
		}
	}
*/
	/*
	if (j < jm && i < im){
		ee[0] = a[0]/(a[0]-1.0f);
		gg[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh)
				 -vf[j_off+i])
				/(a[0]-1.0f);
	}
	*/
/*
	for(k = 1; k < kbm2; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0f/(a[k][j][i]+c[k][j][i]*(1.0f-ee[k-1][j][i])-1.0f);
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (c[k][j][i]*gg[k-1][j][i]-vf[k][j][i])*gg[k][j][i];
			}
		}
	}
*/
	/*
	for (k = 1; k < kbm2; k++){
		if (j < jm && i < im){
			gg[k] = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])-1.0f);

			ee[k] = a[k]*gg[k];

			gg[k] = (c[k]*gg[k-1]-vf[k_off+j_off+i])*gg[k];
		}
	}
	*/
/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			float tmp = 0.25f*(ub[kbm1-1][j][i]+ub[kbm1-1][j][i+1]+ub[kbm1-1][j-1][i]+ub[kbm1-1][j-1][i+1]);
			tps[j][i] = 0.5f*(cbc[j][i]+cbc[j-1][i])
							*sqrtf((tmp*tmp)+(vb[kbm1-1][j][i]*vb[kbm1-1][j][i]));

			vf[kbm1-1][j][i] = (c[kbm1-1][j][i]*gg[kbm2-1][j][i]-vf[kbm1-1][j][i])
								/(tps[j][i]*dti2/(-dz[kbm1-1]*dh[j][i])-1.0f
									-(ee[kbm2-1][j][i]-1.0f)*c[kbm1-1][j][i]);
			vf[kbm1-1][j][i] = vf[kbm1-1][j][i]*dvm[j][i];
		}
	}
*/
	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		float tmp = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
		   	   *sqrtf((tmp*tmp)
		   			 +(vb[kbm1_1_off+j_off+i]
		   				 *vb[kbm1_1_off+j_off+i]));

		vf[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps*dti2
									/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c[kbm1-1]);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];
	}
	*/
/*
	for(ki = kb-3; ki >= 0; ki--){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				vf[ki][j][i] = (ee[ki][j][i]*vf[ki+1][j][i]+gg[ki][j][i])*dvm[j][i];
			}
		}
	}
*/
	/*
	for (ki = kb-3; ki >= 0; ki--){
		if (j < jmm1 && j > 0 && i < imm1 && i > 0){
			vf[ki_off+j_off+i] = (ee[ki]*vf[ki_A1_off+j_off+i]
								 +gg[ki])*dvm[j_off+i];
		}
	}
	*/
/*
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			wvbot[j][i] = -tps[j][i]*vf[kbm1-1][j][i];
		}
	}
*/
	/*
	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		wvbot[j_off+i] = -tps*vf[kbm1_1_off+j_off+i];
	}
	*/


	if (j < jmm1 && j > 0 && i < imm1 && i > 0){
		dh = 0.5f*(h[j_off+i]+etf[j_off+i]
				  +h[j_1_off+i]+etf[j_1_off+i]);

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]
			    +km[jm*im+j_1_off+i])*0.5f;

		a_tmp = -dti2*(c_tmp+umol)
			 	/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh)
				 -vf[j_off+i])
				/(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
			 	/(dz[1]*dzz[0]*dh*dh);

	//for (k = 1; k < kbm1; k++){
	//	if (j < jm && i < im){
	//		a[k-1] = -dti2*(c[k]+umol)
	//			 	/(dz[k-1]*dzz[k-1]*dh*dh);

	//		c[k] = -dti2*(c[k]+umol)
	//			 	/(dz[k]*dzz[k-1]*dh*dh);
	//	}
	//}


		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-vf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tmp_tps = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
		   	   *sqrtf((tmp_tps*tmp_tps)
		   			 +(vb[kbm1_1_off+j_off+i]
		   				 *vb[kbm1_1_off+j_off+i]));

		vf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps*dti2
									/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			vf[ki_off+j_off+i] = (ee[ki]*vf[ki_A1_off+j_off+i]
								 +gg[ki])*dvm[j_off+i];
		}

		wvbot[j_off+i] = -tps*vf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profv_inner_gpu_kernel_0(
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
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y;
	const int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float dh, tps;

	//float a[k_size], c[k_size];
	float ee[k_size], gg[k_size];

	if (j > 32 && i > 32 && j < jm-33 && i < im-33){
		dh = 0.5f*(h[j_off+i]+etf[j_off+i]
				  +h[j_1_off+i]+etf[j_1_off+i]);

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]
			    +km[jm*im+j_1_off+i])*0.5f;

		a_tmp = -dti2*(c_tmp+umol)
			 	/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh)
				 -vf[j_off+i])
				/(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
			 	/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-vf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tmp_tps = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
		   	   *sqrtf((tmp_tps*tmp_tps)
		   			 +(vb[kbm1_1_off+j_off+i]
		   				 *vb[kbm1_1_off+j_off+i]));

		vf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps*dti2
									/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			vf[ki_off+j_off+i] = (ee[ki]*vf[ki_A1_off+j_off+i]
								 +gg[ki])*dvm[j_off+i];
		}

		wvbot[j_off+i] = -tps*vf[kbm1_1_off+j_off+i];
	}
}


__global__ void
profv_ew_gpu_kernel_0(
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
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;
	}

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float dh, tps;

	//float a[k_size], c[k_size];
	float ee[k_size], gg[k_size];

	if (j < jm-1){
		dh = 0.5f*(h[j_off+i]+etf[j_off+i]
				  +h[j_1_off+i]+etf[j_1_off+i]);

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]
			    +km[jm*im+j_1_off+i])*0.5f;

		a_tmp = -dti2*(c_tmp+umol)
			 	/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh)
				 -vf[j_off+i])
				/(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
			 	/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-vf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tmp_tps = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
		   	   *sqrtf((tmp_tps*tmp_tps)
		   			 +(vb[kbm1_1_off+j_off+i]
		   				 *vb[kbm1_1_off+j_off+i]));

		vf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps*dti2
									/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			vf[ki_off+j_off+i] = (ee[ki]*vf[ki_A1_off+j_off+i]
								 +gg[ki])*dvm[j_off+i];
		}

		wvbot[j_off+i] = -tps*vf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profv_sn_gpu_kernel_0(
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
				   int kb, int jm, int im){

	int k, ki;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	int j;
	
	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float dh, tps;

	//float a[k_size], c[k_size];
	float ee[k_size], gg[k_size];

	if (i > 32 && i < im-33){
		dh = 0.5f*(h[j_off+i]+etf[j_off+i]
				  +h[j_1_off+i]+etf[j_1_off+i]);

		float a_tmp, c_tmp, c_tmp_next;

		c_tmp = (km[jm*im+j_off+i]
			    +km[jm*im+j_1_off+i])*0.5f;

		a_tmp = -dti2*(c_tmp+umol)
			 	/(dz[0]*dzz[0]*dh*dh);

		ee[0] = a_tmp/(a_tmp-1.0f);
		gg[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh)
				 -vf[j_off+i])
				/(a_tmp-1.0f);

		c_tmp = -dti2*(c_tmp+umol)
			 	/(dz[1]*dzz[0]*dh*dh);

		for (k = 1; k < kbm2; k++){
			c_tmp_next = (km[k_A1_off+j_off+i]
						 +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k]*dzz[k]*dh*dh);

			gg[k] = 1.0f/(a_tmp+c_tmp*(1.0f-ee[k-1])-1.0f);

			ee[k] = a_tmp*gg[k];

			gg[k] = (c_tmp*gg[k-1]-vf[k_off+j_off+i])*gg[k];

			c_tmp = -dti2*(c_tmp_next+umol)
				 	/(dz[k+1]*dzz[k]*dh*dh);
		}

		float tmp_tps = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		tps = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
		   	   *sqrtf((tmp_tps*tmp_tps)
		   			 +(vb[kbm1_1_off+j_off+i]
		   				 *vb[kbm1_1_off+j_off+i]));

		vf[kbm1_1_off+j_off+i] = (c_tmp*gg[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps*dti2
									/(-dz[kbm1-1]*dh)
								  -1.0f
								  -(ee[kbm2-1]-1.0f)*c_tmp);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		for (ki = kb-3; ki >= 0; ki--){
			vf[ki_off+j_off+i] = (ee[ki]*vf[ki_A1_off+j_off+i]
								 +gg[ki])*dvm[j_off+i];
		}

		wvbot[j_off+i] = -tps*vf[kbm1_1_off+j_off+i];
	}
}

/*
void profv_gpu(float etf[][i_size],float km[][j_size][i_size],
			   float wvsurf[][i_size],float vf[][j_size][i_size],
			   float ub[][j_size][i_size],float vb[][j_size][i_size],
			   float wvbot[][i_size]){
*/

/*
void profv_gpu(float *d_vb, float *d_vf, 
			   float *d_wvsurf, float *d_wvbot,
			   float *d_ub, float *d_km,
			   float *d_etf){
*/
void profv_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_profv,
				   end_profv;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_profv);
#endif

	//int i,j,k,ki;

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	dim3 threadPerBlock_ew(32, 4);
	dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	dim3 threadPerBlock_sn(32, 4);
	dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	//float *d_a = d_3d_tmp0;
	//float *d_c = d_3d_tmp1;
	//float *d_ee = d_3d_tmp2;
	//float *d_gg = d_3d_tmp3;

	//float *d_dh = d_2d_tmp0;
	//float *d_tps = d_2d_tmp1;

	/*
    double a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    double ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
	float dh[j_size][i_size];
	float tps[j_size][i_size];
	*/
	
	/*
	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvsurf, wvsurf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vb, vb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ub, ub, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/

	/*
	profv_gpu_kernel<<<blockPerGrid, threadPerBlock>>>(
			d_dh_profv, d_tps_profv, d_a_profv, d_c_profv, d_ee_profv, d_gg_profv, 
		    d_ub, d_vb, d_vf, d_km, d_etf, d_wvsurf, d_wvbot,
		    d_cbc, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);
	*/
	//profv_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		//d_dh, d_tps, d_a, d_c, d_ee, d_gg, 
	//	    d_ub, d_vb, d_vf, d_km, d_etf, d_wvsurf, d_wvbot,
	//	    d_cbc, d_dvm, d_h, d_dz, d_dzz, 
	//		dti2, umol, kb, jm, im);

/*
#ifdef CUDA_SLICE_MPI
    exchange2d_mpi_gpu(d_wvbot,im,jm);
#else
	checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	
    //exchange2d_mpi_xsz_(wubot,im,jm);
    exchange2d_mpi(wvbot,im,jm);

	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
#endif
*/
    //exchange2d_mpi_gpu(d_wvbot,im,jm);
    //exchange2d_cuda_aware_mpi(d_wvbot,im,jm);

	profv_ew_gpu_kernel_0<<<blockPerGrid_ew, threadPerBlock_ew,
						    0, stream[1]>>>(
		    d_ub, d_vb, d_vf, d_km, d_etf, d_wvsurf, d_wvbot,
		    d_cbc, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);

	profv_sn_gpu_kernel_0<<<blockPerGrid_sn, threadPerBlock_sn,
							0, stream[2]>>>(
		    d_ub, d_vb, d_vf, d_km, d_etf, d_wvsurf, d_wvbot,
		    d_cbc, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);

	profv_inner_gpu_kernel_0<<<blockPerGrid, threadPerBlock,
							0, stream[0]>>>(
		    d_ub, d_vb, d_vf, d_km, d_etf, d_wvsurf, d_wvbot,
		    d_cbc, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

    //exchange2d_mpi_gpu(d_wvbot, im,jm);

    exchange2d_cudaUVA(d_wvbot,d_wvbot_east, d_wvbot_west,
					   d_wvbot_south, d_wvbot_north,
					   stream[1], im,jm);

	//MPI_Barrier(pom_comm);
    //exchange2d_cuda_ipc(d_wvbot,d_wvbot_east, d_wvbot_west,
	//					stream[1], im,jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[0]));

	
	/*
	checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(wvbot, d_wvbot, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_profv);
		profv_time += time_consumed(&start_profv, 
									&end_profv);
#endif
	
}


__global__ void
vort_gpu_kernel_0(float * __restrict__ fx_ctsurf, 
		          float * __restrict__ fy_ctsurf, 
				  float * __restrict__ fx_ctbot, 
				  float * __restrict__ fy_ctbot, 
				  float * __restrict__ fx_celg, 
				  float * __restrict__ fy_celg, 
				  float * __restrict__ fx_cjbar, 
				  float * __restrict__ fy_cjbar, 
				  float * __restrict__ fx_cadv, 
				  float * __restrict__ fy_cadv,
				  float * __restrict__ fx_cpvf, 
				  float * __restrict__ fy_cpvf,
				  float * __restrict__ fx_cten, 
				  float * __restrict__ fy_cten,
				  const float * __restrict__ ctsurf, 
				  const float * __restrict__ ctbot, 
				  const float * __restrict__ cpvf, 
				  const float * __restrict__ cjbar,
				  const float * __restrict__ cadv, 
				  const float * __restrict__ cten, 
				  const float * __restrict__ celg,
				  float * __restrict__ totx, 
				  float * __restrict__ toty,
				  const float * __restrict__ d, 
				  const float * __restrict__ elb, 
				  const float * __restrict__ el, 
				  const float * __restrict__ elf,
				  const float * __restrict__ drx2d, 
				  const float * __restrict__ dry2d,
				  const float * __restrict__ adx2d, 
				  const float * __restrict__ ady2d, 
				  const float * __restrict__ advua, 
				  const float * __restrict__ advva, 
				  const float * __restrict__ uab, 
				  const float * __restrict__ vab, 
				  const float * __restrict__ ua, 
				  const float * __restrict__ va, 
				  const float * __restrict__ uaf, 
				  const float * __restrict__ vaf, 
				  const float * __restrict__ wusurf, 
				  const float * __restrict__ wvsurf, 
				  const float * __restrict__ wubot, 
				  const float * __restrict__ wvbot, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv,
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ h,
		          float grav, float dte,
				  int iid, int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	float dmx, dmy;
	float totx_tmp = 0, 
		  toty_tmp = 0;

	float alpha = 0.225f;

	if (i > 0 && i < im &&
		j > 0 && j < jm){
		
		if (iid == 1){
			dmx = 0.5f*(d[j_off+i]+d[j_off+i-1]);	
			dmy = 0.5f*(d[j_off+i]+d[j_1_off+i]);	
		}else{
			dmx = 1.f;
			dmy = 1.f;
		}

		fx_ctsurf[j_off+i] = wusurf[j_off+i]/dmx;
		fy_ctsurf[j_off+i] = wvsurf[j_off+i]/dmy;

		totx_tmp += fx_ctsurf[j_off+i];
		toty_tmp += fy_ctsurf[j_off+i];

		fx_ctbot[j_off+i] = -wubot[j_off+i]/dmx;
		fy_ctbot[j_off+i] = -wvbot[j_off+i]/dmy;

		totx_tmp += fx_ctbot[j_off+i];
		toty_tmp += fy_ctbot[j_off+i];

		fx_celg[j_off+i] = 0.25f*grav*(dy[j_off+i]+dy[j_off+i-1])
								*(d[j_off+i]+d[j_off+i-1])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_off+i-1])
				  				  +alpha*(elb[j_off+i]-elb[j_off+i-1]
										 +elf[j_off+i]-elf[j_off+i-1]));

		fx_celg[j_off+i] = fx_celg[j_off+i]/(aru[j_off+i]*dmx);

		fy_celg[j_off+i] = 0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
								*(d[j_off+i]+d[j_1_off+i])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_1_off+i])
				  				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
										 +elf[j_off+i]-elf[j_1_off+i]));

		fy_celg[j_off+i] = fy_celg[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_celg[j_off+i];
		toty_tmp += fy_celg[j_off+i];

		fx_cjbar[j_off+i] = drx2d[j_off+i]/(aru[j_off+i]*dmx);
		fy_cjbar[j_off+i] = dry2d[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_cjbar[j_off+i];
		toty_tmp += fy_cjbar[j_off+i];

		fx_cadv[j_off+i] = (adx2d[j_off+i]+advua[j_off+i])/(aru[j_off+i]*dmx);
		fy_cadv[j_off+i] = (ady2d[j_off+i]+advva[j_off+i])/(arv[j_off+i]*dmy);

		totx_tmp += fx_cadv[j_off+i];
		toty_tmp += fy_cadv[j_off+i];

		fx_cpvf[j_off+i] = -0.25f*(cor[j_off+i]*d[j_off+i]
							*(va[j_A1_off+i]+va[j_off+i])
						  +cor[j_off+i-1]*d[j_off+i-1]
						    *(va[j_A1_off+i-1]+va[j_off+i-1]))/dmx;

		fy_cpvf[j_off+i] = 0.25f*(cor[j_off+i]*d[j_off+i]
							*(ua[j_off+i+1]+ua[j_off+i])
						 +cor[j_1_off+i]*d[j_1_off+i]
							*(ua[j_1_off+i+1]+ua[j_1_off+i]))/dmy;

		totx_tmp += fx_cpvf[j_off+i];
		toty_tmp += fy_cpvf[j_off+i];


		fx_cten[j_off+i] = (uaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_off+i-1]+elf[j_off+i-1])
				    -uab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							   +h[j_off+i-1]+elb[j_off+i-1]))
				   /(4.f*dte*dmx)*dum[j_off+i];

		fy_cten[j_off+i] = (vaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_1_off+i]+elf[j_1_off+i])
					 -vab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							    +h[j_1_off+i]+elb[j_1_off+i]))
				   /(4.f*dte*dmy)*dvm[j_off+i];

		totx_tmp += fx_cten[j_off+i];
		toty_tmp += fy_cten[j_off+i];

		totx_tmp = totx_tmp*dum[j_off+i]*dum[j_off+i-1];
		toty_tmp = totx_tmp*dvm[j_off+i]*dvm[j_1_off+i];

		totx[j_off+i] = totx_tmp;
		toty[j_off+i] = toty_tmp;
	}
}

__global__ void
vort_inner_gpu_kernel_0(float * __restrict__ fx_ctsurf, 
		          float * __restrict__ fy_ctsurf, 
				  float * __restrict__ fx_ctbot, 
				  float * __restrict__ fy_ctbot, 
				  float * __restrict__ fx_celg, 
				  float * __restrict__ fy_celg, 
				  float * __restrict__ fx_cjbar, 
				  float * __restrict__ fy_cjbar, 
				  float * __restrict__ fx_cadv, 
				  float * __restrict__ fy_cadv,
				  float * __restrict__ fx_cpvf, 
				  float * __restrict__ fy_cpvf,
				  float * __restrict__ fx_cten, 
				  float * __restrict__ fy_cten,
				  const float * __restrict__ ctsurf, 
				  const float * __restrict__ ctbot, 
				  const float * __restrict__ cpvf, 
				  const float * __restrict__ cjbar,
				  const float * __restrict__ cadv, 
				  const float * __restrict__ cten, 
				  const float * __restrict__ celg,
				  float * __restrict__ totx, 
				  float * __restrict__ toty,
				  const float * __restrict__ d, 
				  const float * __restrict__ elb, 
				  const float * __restrict__ el, 
				  const float * __restrict__ elf,
				  const float * __restrict__ drx2d, 
				  const float * __restrict__ dry2d,
				  const float * __restrict__ adx2d, 
				  const float * __restrict__ ady2d, 
				  const float * __restrict__ advua, 
				  const float * __restrict__ advva, 
				  const float * __restrict__ uab, 
				  const float * __restrict__ vab, 
				  const float * __restrict__ ua, 
				  const float * __restrict__ va, 
				  const float * __restrict__ uaf, 
				  const float * __restrict__ vaf, 
				  const float * __restrict__ wusurf, 
				  const float * __restrict__ wvsurf, 
				  const float * __restrict__ wubot, 
				  const float * __restrict__ wvbot, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv,
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ h,
		          float grav, float dte,
				  int iid, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1;

	float dmx, dmy;
	float totx_tmp = 0, 
		  toty_tmp = 0;

	const float alpha = 0.225f;

	if (i > 32 && j > 32 && i < im-33 && j < jm-33){
		
		if (iid == 1){
			dmx = 0.5f*(d[j_off+i]+d[j_off+i-1]);	
			dmy = 0.5f*(d[j_off+i]+d[j_1_off+i]);	
		}else{
			dmx = 1.f;
			dmy = 1.f;
		}

		fx_ctsurf[j_off+i] = wusurf[j_off+i]/dmx;
		fy_ctsurf[j_off+i] = wvsurf[j_off+i]/dmy;

		totx_tmp += fx_ctsurf[j_off+i];
		toty_tmp += fy_ctsurf[j_off+i];

		fx_ctbot[j_off+i] = -wubot[j_off+i]/dmx;
		fy_ctbot[j_off+i] = -wvbot[j_off+i]/dmy;

		totx_tmp += fx_ctbot[j_off+i];
		toty_tmp += fy_ctbot[j_off+i];

		fx_celg[j_off+i] = 0.25f*grav*(dy[j_off+i]+dy[j_off+i-1])
								*(d[j_off+i]+d[j_off+i-1])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_off+i-1])
				  				  +alpha*(elb[j_off+i]-elb[j_off+i-1]
										 +elf[j_off+i]-elf[j_off+i-1]));

		fx_celg[j_off+i] = fx_celg[j_off+i]/(aru[j_off+i]*dmx);

		fy_celg[j_off+i] = 0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
								*(d[j_off+i]+d[j_1_off+i])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_1_off+i])
				  				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
										 +elf[j_off+i]-elf[j_1_off+i]));

		fy_celg[j_off+i] = fy_celg[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_celg[j_off+i];
		toty_tmp += fy_celg[j_off+i];

		fx_cjbar[j_off+i] = drx2d[j_off+i]/(aru[j_off+i]*dmx);
		fy_cjbar[j_off+i] = dry2d[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_cjbar[j_off+i];
		toty_tmp += fy_cjbar[j_off+i];

		fx_cadv[j_off+i] = (adx2d[j_off+i]+advua[j_off+i])/(aru[j_off+i]*dmx);
		fy_cadv[j_off+i] = (ady2d[j_off+i]+advva[j_off+i])/(arv[j_off+i]*dmy);

		totx_tmp += fx_cadv[j_off+i];
		toty_tmp += fy_cadv[j_off+i];

		fx_cpvf[j_off+i] = -0.25f*(cor[j_off+i]*d[j_off+i]
							*(va[j_A1_off+i]+va[j_off+i])
						  +cor[j_off+i-1]*d[j_off+i-1]
						    *(va[j_A1_off+i-1]+va[j_off+i-1]))/dmx;

		fy_cpvf[j_off+i] = 0.25f*(cor[j_off+i]*d[j_off+i]
							*(ua[j_off+i+1]+ua[j_off+i])
						 +cor[j_1_off+i]*d[j_1_off+i]
							*(ua[j_1_off+i+1]+ua[j_1_off+i]))/dmy;

		totx_tmp += fx_cpvf[j_off+i];
		toty_tmp += fy_cpvf[j_off+i];


		fx_cten[j_off+i] = (uaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_off+i-1]+elf[j_off+i-1])
				    -uab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							   +h[j_off+i-1]+elb[j_off+i-1]))
				   /(4.f*dte*dmx)*dum[j_off+i];

		fy_cten[j_off+i] = (vaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_1_off+i]+elf[j_1_off+i])
					 -vab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							    +h[j_1_off+i]+elb[j_1_off+i]))
				   /(4.f*dte*dmy)*dvm[j_off+i];

		totx_tmp += fx_cten[j_off+i];
		toty_tmp += fy_cten[j_off+i];

		totx_tmp = totx_tmp*dum[j_off+i]*dum[j_off+i-1];
		toty_tmp = totx_tmp*dvm[j_off+i]*dvm[j_1_off+i];

		totx[j_off+i] = totx_tmp;
		toty[j_off+i] = toty_tmp;
	}
}

__global__ void
vort_ew_gpu_kernel_0(float * __restrict__ fx_ctsurf, 
		          float * __restrict__ fy_ctsurf, 
				  float * __restrict__ fx_ctbot, 
				  float * __restrict__ fy_ctbot, 
				  float * __restrict__ fx_celg, 
				  float * __restrict__ fy_celg, 
				  float * __restrict__ fx_cjbar, 
				  float * __restrict__ fy_cjbar, 
				  float * __restrict__ fx_cadv, 
				  float * __restrict__ fy_cadv,
				  float * __restrict__ fx_cpvf, 
				  float * __restrict__ fy_cpvf,
				  float * __restrict__ fx_cten, 
				  float * __restrict__ fy_cten,
				  const float * __restrict__ ctsurf, 
				  const float * __restrict__ ctbot, 
				  const float * __restrict__ cpvf, 
				  const float * __restrict__ cjbar,
				  const float * __restrict__ cadv, 
				  const float * __restrict__ cten, 
				  const float * __restrict__ celg,
				  float * __restrict__ totx, 
				  float * __restrict__ toty,
				  const float * __restrict__ d, 
				  const float * __restrict__ elb, 
				  const float * __restrict__ el, 
				  const float * __restrict__ elf,
				  const float * __restrict__ drx2d, 
				  const float * __restrict__ dry2d,
				  const float * __restrict__ adx2d, 
				  const float * __restrict__ ady2d, 
				  const float * __restrict__ advua, 
				  const float * __restrict__ advva, 
				  const float * __restrict__ uab, 
				  const float * __restrict__ vab, 
				  const float * __restrict__ ua, 
				  const float * __restrict__ va, 
				  const float * __restrict__ uaf, 
				  const float * __restrict__ vaf, 
				  const float * __restrict__ wusurf, 
				  const float * __restrict__ wvsurf, 
				  const float * __restrict__ wubot, 
				  const float * __restrict__ wvbot, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv,
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ h,
		          float grav, float dte,
				  int iid, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1;
	int i;

	float dmx, dmy;
	float totx_tmp = 0, 
		  toty_tmp = 0;

	const float alpha = 0.225f;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;
	}

	if (j < jm-1){
		
		if (iid == 1){
			dmx = 0.5f*(d[j_off+i]+d[j_off+i-1]);	
			dmy = 0.5f*(d[j_off+i]+d[j_1_off+i]);	
		}else{
			dmx = 1.f;
			dmy = 1.f;
		}

		fx_ctsurf[j_off+i] = wusurf[j_off+i]/dmx;
		fy_ctsurf[j_off+i] = wvsurf[j_off+i]/dmy;

		totx_tmp += fx_ctsurf[j_off+i];
		toty_tmp += fy_ctsurf[j_off+i];

		fx_ctbot[j_off+i] = -wubot[j_off+i]/dmx;
		fy_ctbot[j_off+i] = -wvbot[j_off+i]/dmy;

		totx_tmp += fx_ctbot[j_off+i];
		toty_tmp += fy_ctbot[j_off+i];

		fx_celg[j_off+i] = 0.25f*grav*(dy[j_off+i]+dy[j_off+i-1])
								*(d[j_off+i]+d[j_off+i-1])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_off+i-1])
				  				  +alpha*(elb[j_off+i]-elb[j_off+i-1]
										 +elf[j_off+i]-elf[j_off+i-1]));

		fx_celg[j_off+i] = fx_celg[j_off+i]/(aru[j_off+i]*dmx);

		fy_celg[j_off+i] = 0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
								*(d[j_off+i]+d[j_1_off+i])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_1_off+i])
				  				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
										 +elf[j_off+i]-elf[j_1_off+i]));

		fy_celg[j_off+i] = fy_celg[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_celg[j_off+i];
		toty_tmp += fy_celg[j_off+i];

		fx_cjbar[j_off+i] = drx2d[j_off+i]/(aru[j_off+i]*dmx);
		fy_cjbar[j_off+i] = dry2d[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_cjbar[j_off+i];
		toty_tmp += fy_cjbar[j_off+i];

		fx_cadv[j_off+i] = (adx2d[j_off+i]+advua[j_off+i])/(aru[j_off+i]*dmx);
		fy_cadv[j_off+i] = (ady2d[j_off+i]+advva[j_off+i])/(arv[j_off+i]*dmy);

		totx_tmp += fx_cadv[j_off+i];
		toty_tmp += fy_cadv[j_off+i];

		fx_cpvf[j_off+i] = -0.25f*(cor[j_off+i]*d[j_off+i]
							*(va[j_A1_off+i]+va[j_off+i])
						  +cor[j_off+i-1]*d[j_off+i-1]
						    *(va[j_A1_off+i-1]+va[j_off+i-1]))/dmx;

		fy_cpvf[j_off+i] = 0.25f*(cor[j_off+i]*d[j_off+i]
							*(ua[j_off+i+1]+ua[j_off+i])
						 +cor[j_1_off+i]*d[j_1_off+i]
							*(ua[j_1_off+i+1]+ua[j_1_off+i]))/dmy;

		totx_tmp += fx_cpvf[j_off+i];
		toty_tmp += fy_cpvf[j_off+i];


		fx_cten[j_off+i] = (uaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_off+i-1]+elf[j_off+i-1])
				    -uab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							   +h[j_off+i-1]+elb[j_off+i-1]))
				   /(4.f*dte*dmx)*dum[j_off+i];

		fy_cten[j_off+i] = (vaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_1_off+i]+elf[j_1_off+i])
					 -vab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							    +h[j_1_off+i]+elb[j_1_off+i]))
				   /(4.f*dte*dmy)*dvm[j_off+i];

		totx_tmp += fx_cten[j_off+i];
		toty_tmp += fy_cten[j_off+i];

		totx_tmp = totx_tmp*dum[j_off+i]*dum[j_off+i-1];
		toty_tmp = totx_tmp*dvm[j_off+i]*dvm[j_1_off+i];

		totx[j_off+i] = totx_tmp;
		toty[j_off+i] = toty_tmp;
	}
}

__global__ void
vort_ew_bcond_gpu_kernel_0(float * __restrict__ fx_ctsurf, 
		          float * __restrict__ fy_ctsurf, 
				  float * __restrict__ fx_ctbot, 
				  float * __restrict__ fy_ctbot, 
				  float * __restrict__ fx_celg, 
				  float * __restrict__ fy_celg, 
				  float * __restrict__ fx_cjbar, 
				  float * __restrict__ fy_cjbar, 
				  float * __restrict__ fx_cadv, 
				  float * __restrict__ fy_cadv,
				  float * __restrict__ fx_cpvf, 
				  float * __restrict__ fy_cpvf,
				  float * __restrict__ fx_cten, 
				  float * __restrict__ fy_cten,
				  const float * __restrict__ ctsurf, 
				  const float * __restrict__ ctbot, 
				  const float * __restrict__ cpvf, 
				  const float * __restrict__ cjbar,
				  const float * __restrict__ cadv, 
				  const float * __restrict__ cten, 
				  const float * __restrict__ celg,
				  float * __restrict__ totx, 
				  float * __restrict__ toty,
				  const float * __restrict__ d, 
				  const float * __restrict__ elb, 
				  const float * __restrict__ el, 
				  const float * __restrict__ elf,
				  const float * __restrict__ drx2d, 
				  const float * __restrict__ dry2d,
				  const float * __restrict__ adx2d, 
				  const float * __restrict__ ady2d, 
				  const float * __restrict__ advua, 
				  const float * __restrict__ advva, 
				  const float * __restrict__ uab, 
				  const float * __restrict__ vab, 
				  const float * __restrict__ ua, 
				  const float * __restrict__ va, 
				  const float * __restrict__ uaf, 
				  const float * __restrict__ vaf, 
				  const float * __restrict__ wusurf, 
				  const float * __restrict__ wvsurf, 
				  const float * __restrict__ wubot, 
				  const float * __restrict__ wvbot, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv,
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ h,
				  float grav, 
				  float dte,
		          int iid, int n_west, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+1;
	const int i = im-1;

	float dmx, dmy;
	float totx_tmp = 0, 
		  toty_tmp = 0;

	const float alpha = 0.225f;

	if (j < jm){
		
		if (iid == 1){
			dmx = 0.5f*(d[j_off+i]+d[j_off+i-1]);	
			dmy = 0.5f*(d[j_off+i]+d[j_1_off+i]);	
		}else{
			dmx = 1.f;
			dmy = 1.f;
		}

		fx_ctsurf[j_off+i] = wusurf[j_off+i]/dmx;
		fy_ctsurf[j_off+i] = wvsurf[j_off+i]/dmy;

		totx_tmp += fx_ctsurf[j_off+i];
		toty_tmp += fy_ctsurf[j_off+i];

		fx_ctbot[j_off+i] = -wubot[j_off+i]/dmx;
		fy_ctbot[j_off+i] = -wvbot[j_off+i]/dmy;

		totx_tmp += fx_ctbot[j_off+i];
		toty_tmp += fy_ctbot[j_off+i];

		fx_celg[j_off+i] = 0.25f*grav*(dy[j_off+i]+dy[j_off+i-1])
								*(d[j_off+i]+d[j_off+i-1])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_off+i-1])
				  				  +alpha*(elb[j_off+i]-elb[j_off+i-1]
										 +elf[j_off+i]-elf[j_off+i-1]));

		fx_celg[j_off+i] = fx_celg[j_off+i]/(aru[j_off+i]*dmx);

		fy_celg[j_off+i] = 0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
								*(d[j_off+i]+d[j_1_off+i])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_1_off+i])
				  				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
										 +elf[j_off+i]-elf[j_1_off+i]));

		fy_celg[j_off+i] = fy_celg[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_celg[j_off+i];
		toty_tmp += fy_celg[j_off+i];

		fx_cjbar[j_off+i] = drx2d[j_off+i]/(aru[j_off+i]*dmx);
		fy_cjbar[j_off+i] = dry2d[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_cjbar[j_off+i];
		toty_tmp += fy_cjbar[j_off+i];

		fx_cadv[j_off+i] = (adx2d[j_off+i]+advua[j_off+i])/(aru[j_off+i]*dmx);
		fy_cadv[j_off+i] = (ady2d[j_off+i]+advva[j_off+i])/(arv[j_off+i]*dmy);

		totx_tmp += fx_cadv[j_off+i];
		toty_tmp += fy_cadv[j_off+i];

		fx_cpvf[j_off+i] = -0.25f*(cor[j_off+i]*d[j_off+i]
							*(va[j_A1_off+i]+va[j_off+i])
						  +cor[j_off+i-1]*d[j_off+i-1]
						    *(va[j_A1_off+i-1]+va[j_off+i-1]))/dmx;

		fy_cpvf[j_off+i] = 0.25f*(cor[j_off+i]*d[j_off+i]
							*(ua[j_off+i+1]+ua[j_off+i])
						 +cor[j_1_off+i]*d[j_1_off+i]
							*(ua[j_1_off+i+1]+ua[j_1_off+i]))/dmy;

		totx_tmp += fx_cpvf[j_off+i];
		toty_tmp += fy_cpvf[j_off+i];


		fx_cten[j_off+i] = (uaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_off+i-1]+elf[j_off+i-1])
				    -uab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							   +h[j_off+i-1]+elb[j_off+i-1]))
				   /(4.f*dte*dmx)*dum[j_off+i];

		fy_cten[j_off+i] = (vaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_1_off+i]+elf[j_1_off+i])
					 -vab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							    +h[j_1_off+i]+elb[j_1_off+i]))
				   /(4.f*dte*dmy)*dvm[j_off+i];

		totx_tmp += fx_cten[j_off+i];
		toty_tmp += fy_cten[j_off+i];

		totx_tmp = totx_tmp*dum[j_off+i]*dum[j_off+i-1];
		toty_tmp = totx_tmp*dvm[j_off+i]*dvm[j_1_off+i];

		totx[j_off+i] = totx_tmp;
		toty[j_off+i] = toty_tmp;
	}

	if (n_west == -1){
		if (j < jm-1){
			fy_ctsurf[j_off] = 0;

			fy_ctbot[j_off] = 0;

			fy_celg[j_off] = 0;

			fy_cjbar[j_off] = 0;

			fy_cadv[j_off] = 0;

			fy_cpvf[j_off] = 0;

			fy_cten[j_off] = 0;
		}
	
	}
}

__global__ void
vort_sn_gpu_kernel_0(float * __restrict__ fx_ctsurf, 
		          float * __restrict__ fy_ctsurf, 
				  float * __restrict__ fx_ctbot, 
				  float * __restrict__ fy_ctbot, 
				  float * __restrict__ fx_celg, 
				  float * __restrict__ fy_celg, 
				  float * __restrict__ fx_cjbar, 
				  float * __restrict__ fy_cjbar, 
				  float * __restrict__ fx_cadv, 
				  float * __restrict__ fy_cadv,
				  float * __restrict__ fx_cpvf, 
				  float * __restrict__ fy_cpvf,
				  float * __restrict__ fx_cten, 
				  float * __restrict__ fy_cten,
				  const float * __restrict__ ctsurf, 
				  const float * __restrict__ ctbot, 
				  const float * __restrict__ cpvf, 
				  const float * __restrict__ cjbar,
				  const float * __restrict__ cadv, 
				  const float * __restrict__ cten, 
				  const float * __restrict__ celg,
				  float * __restrict__ totx, 
				  float * __restrict__ toty,
				  const float * __restrict__ d, 
				  const float * __restrict__ elb, 
				  const float * __restrict__ el, 
				  const float * __restrict__ elf,
				  const float * __restrict__ drx2d, 
				  const float * __restrict__ dry2d,
				  const float * __restrict__ adx2d, 
				  const float * __restrict__ ady2d, 
				  const float * __restrict__ advua, 
				  const float * __restrict__ advva, 
				  const float * __restrict__ uab, 
				  const float * __restrict__ vab, 
				  const float * __restrict__ ua, 
				  const float * __restrict__ va, 
				  const float * __restrict__ uaf, 
				  const float * __restrict__ vaf, 
				  const float * __restrict__ wusurf, 
				  const float * __restrict__ wvsurf, 
				  const float * __restrict__ wubot, 
				  const float * __restrict__ wvbot, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv,
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ h,
		          float grav, float dte,
				  int iid, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1;
	int j;

	float dmx, dmy;
	float totx_tmp = 0, 
		  toty_tmp = 0;

	const float alpha = 0.225f;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){
		if (iid == 1){
			dmx = 0.5f*(d[j_off+i]+d[j_off+i-1]);	
			dmy = 0.5f*(d[j_off+i]+d[j_1_off+i]);	
		}else{
			dmx = 1.f;
			dmy = 1.f;
		}

		fx_ctsurf[j_off+i] = wusurf[j_off+i]/dmx;
		fy_ctsurf[j_off+i] = wvsurf[j_off+i]/dmy;

		totx_tmp += fx_ctsurf[j_off+i];
		toty_tmp += fy_ctsurf[j_off+i];

		fx_ctbot[j_off+i] = -wubot[j_off+i]/dmx;
		fy_ctbot[j_off+i] = -wvbot[j_off+i]/dmy;

		totx_tmp += fx_ctbot[j_off+i];
		toty_tmp += fy_ctbot[j_off+i];

		fx_celg[j_off+i] = 0.25f*grav*(dy[j_off+i]+dy[j_off+i-1])
								*(d[j_off+i]+d[j_off+i-1])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_off+i-1])
				  				  +alpha*(elb[j_off+i]-elb[j_off+i-1]
										 +elf[j_off+i]-elf[j_off+i-1]));

		fx_celg[j_off+i] = fx_celg[j_off+i]/(aru[j_off+i]*dmx);

		fy_celg[j_off+i] = 0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
								*(d[j_off+i]+d[j_1_off+i])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_1_off+i])
				  				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
										 +elf[j_off+i]-elf[j_1_off+i]));

		fy_celg[j_off+i] = fy_celg[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_celg[j_off+i];
		toty_tmp += fy_celg[j_off+i];

		fx_cjbar[j_off+i] = drx2d[j_off+i]/(aru[j_off+i]*dmx);
		fy_cjbar[j_off+i] = dry2d[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_cjbar[j_off+i];
		toty_tmp += fy_cjbar[j_off+i];

		fx_cadv[j_off+i] = (adx2d[j_off+i]+advua[j_off+i])/(aru[j_off+i]*dmx);
		fy_cadv[j_off+i] = (ady2d[j_off+i]+advva[j_off+i])/(arv[j_off+i]*dmy);

		totx_tmp += fx_cadv[j_off+i];
		toty_tmp += fy_cadv[j_off+i];

		fx_cpvf[j_off+i] = -0.25f*(cor[j_off+i]*d[j_off+i]
							*(va[j_A1_off+i]+va[j_off+i])
						  +cor[j_off+i-1]*d[j_off+i-1]
						    *(va[j_A1_off+i-1]+va[j_off+i-1]))/dmx;

		fy_cpvf[j_off+i] = 0.25f*(cor[j_off+i]*d[j_off+i]
							*(ua[j_off+i+1]+ua[j_off+i])
						 +cor[j_1_off+i]*d[j_1_off+i]
							*(ua[j_1_off+i+1]+ua[j_1_off+i]))/dmy;

		totx_tmp += fx_cpvf[j_off+i];
		toty_tmp += fy_cpvf[j_off+i];


		fx_cten[j_off+i] = (uaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_off+i-1]+elf[j_off+i-1])
				    -uab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							   +h[j_off+i-1]+elb[j_off+i-1]))
				   /(4.f*dte*dmx)*dum[j_off+i];

		fy_cten[j_off+i] = (vaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_1_off+i]+elf[j_1_off+i])
					 -vab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							    +h[j_1_off+i]+elb[j_1_off+i]))
				   /(4.f*dte*dmy)*dvm[j_off+i];

		totx_tmp += fx_cten[j_off+i];
		toty_tmp += fy_cten[j_off+i];

		totx_tmp = totx_tmp*dum[j_off+i]*dum[j_off+i-1];
		toty_tmp = totx_tmp*dvm[j_off+i]*dvm[j_1_off+i];

		totx[j_off+i] = totx_tmp;
		toty[j_off+i] = toty_tmp;
	}
}

__global__ void
vort_sn_bcond_gpu_kernel_0(float * __restrict__ fx_ctsurf, 
		          float * __restrict__ fy_ctsurf, 
				  float * __restrict__ fx_ctbot, 
				  float * __restrict__ fy_ctbot, 
				  float * __restrict__ fx_celg, 
				  float * __restrict__ fy_celg, 
				  float * __restrict__ fx_cjbar, 
				  float * __restrict__ fy_cjbar, 
				  float * __restrict__ fx_cadv, 
				  float * __restrict__ fy_cadv,
				  float * __restrict__ fx_cpvf, 
				  float * __restrict__ fy_cpvf,
				  float * __restrict__ fx_cten, 
				  float * __restrict__ fy_cten,
				  const float * __restrict__ ctsurf, 
				  const float * __restrict__ ctbot, 
				  const float * __restrict__ cpvf, 
				  const float * __restrict__ cjbar,
				  const float * __restrict__ cadv, 
				  const float * __restrict__ cten, 
				  const float * __restrict__ celg,
				  float * __restrict__ totx, 
				  float * __restrict__ toty,
				  const float * __restrict__ d, 
				  const float * __restrict__ elb, 
				  const float * __restrict__ el, 
				  const float * __restrict__ elf,
				  const float * __restrict__ drx2d, 
				  const float * __restrict__ dry2d,
				  const float * __restrict__ adx2d, 
				  const float * __restrict__ ady2d, 
				  const float * __restrict__ advua, 
				  const float * __restrict__ advva, 
				  const float * __restrict__ uab, 
				  const float * __restrict__ vab, 
				  const float * __restrict__ ua, 
				  const float * __restrict__ va, 
				  const float * __restrict__ uaf, 
				  const float * __restrict__ vaf, 
				  const float * __restrict__ wusurf, 
				  const float * __restrict__ wvsurf, 
				  const float * __restrict__ wubot, 
				  const float * __restrict__ wvbot, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv,
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  const float * __restrict__ cor, 
				  const float * __restrict__ h,
		          float grav, float dte,
				  int iid, int n_south, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x+1;
	const int j = jm-1;

	float dmx, dmy;
	float totx_tmp = 0, 
		  toty_tmp = 0;

	const float alpha = 0.225f;

	if (i < im-1){
		if (iid == 1){
			dmx = 0.5f*(d[j_off+i]+d[j_off+i-1]);	
			dmy = 0.5f*(d[j_off+i]+d[j_1_off+i]);	
		}else{
			dmx = 1.f;
			dmy = 1.f;
		}

		fx_ctsurf[j_off+i] = wusurf[j_off+i]/dmx;
		fy_ctsurf[j_off+i] = wvsurf[j_off+i]/dmy;

		totx_tmp += fx_ctsurf[j_off+i];
		toty_tmp += fy_ctsurf[j_off+i];

		fx_ctbot[j_off+i] = -wubot[j_off+i]/dmx;
		fy_ctbot[j_off+i] = -wvbot[j_off+i]/dmy;

		totx_tmp += fx_ctbot[j_off+i];
		toty_tmp += fy_ctbot[j_off+i];

		fx_celg[j_off+i] = 0.25f*grav*(dy[j_off+i]+dy[j_off+i-1])
								*(d[j_off+i]+d[j_off+i-1])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_off+i-1])
				  				  +alpha*(elb[j_off+i]-elb[j_off+i-1]
										 +elf[j_off+i]-elf[j_off+i-1]));

		fx_celg[j_off+i] = fx_celg[j_off+i]/(aru[j_off+i]*dmx);

		fy_celg[j_off+i] = 0.25f*grav*(dx[j_off+i]+dx[j_1_off+i])
								*(d[j_off+i]+d[j_1_off+i])
				  				*((1.f-2.f*alpha)*(el[j_off+i]-el[j_1_off+i])
				  				  +alpha*(elb[j_off+i]-elb[j_1_off+i]
										 +elf[j_off+i]-elf[j_1_off+i]));

		fy_celg[j_off+i] = fy_celg[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_celg[j_off+i];
		toty_tmp += fy_celg[j_off+i];

		fx_cjbar[j_off+i] = drx2d[j_off+i]/(aru[j_off+i]*dmx);
		fy_cjbar[j_off+i] = dry2d[j_off+i]/(arv[j_off+i]*dmy);

		totx_tmp += fx_cjbar[j_off+i];
		toty_tmp += fy_cjbar[j_off+i];

		fx_cadv[j_off+i] = (adx2d[j_off+i]+advua[j_off+i])/(aru[j_off+i]*dmx);
		fy_cadv[j_off+i] = (ady2d[j_off+i]+advva[j_off+i])/(arv[j_off+i]*dmy);

		totx_tmp += fx_cadv[j_off+i];
		toty_tmp += fy_cadv[j_off+i];

		fx_cpvf[j_off+i] = -0.25f*(cor[j_off+i]*d[j_off+i]
							*(va[j_A1_off+i]+va[j_off+i])
						  +cor[j_off+i-1]*d[j_off+i-1]
						    *(va[j_A1_off+i-1]+va[j_off+i-1]))/dmx;

		fy_cpvf[j_off+i] = 0.25f*(cor[j_off+i]*d[j_off+i]
							*(ua[j_off+i+1]+ua[j_off+i])
						 +cor[j_1_off+i]*d[j_1_off+i]
							*(ua[j_1_off+i+1]+ua[j_1_off+i]))/dmy;

		totx_tmp += fx_cpvf[j_off+i];
		toty_tmp += fy_cpvf[j_off+i];


		fx_cten[j_off+i] = (uaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_off+i-1]+elf[j_off+i-1])
				    -uab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							   +h[j_off+i-1]+elb[j_off+i-1]))
				   /(4.f*dte*dmx)*dum[j_off+i];

		fy_cten[j_off+i] = (vaf[j_off+i]*(h[j_off+i]+elf[j_off+i]
							  +h[j_1_off+i]+elf[j_1_off+i])
					 -vab[j_off+i]*(h[j_off+i]+elb[j_off+i]
							    +h[j_1_off+i]+elb[j_1_off+i]))
				   /(4.f*dte*dmy)*dvm[j_off+i];

		totx_tmp += fx_cten[j_off+i];
		toty_tmp += fy_cten[j_off+i];

		totx_tmp = totx_tmp*dum[j_off+i]*dum[j_off+i-1];
		toty_tmp = totx_tmp*dvm[j_off+i]*dvm[j_1_off+i];

		totx[j_off+i] = totx_tmp;
		toty[j_off+i] = toty_tmp;
	}

	if (n_south == -1){
		fx_ctsurf[i] = 0;

		fx_ctbot[i] = 0;

		fx_celg[i] = 0;

		fx_cjbar[i] = 0;

		fx_cadv[i] = 0;

		fx_cpvf[i] = 0;

		fx_cten[i] = 0;
	}
}

__global__ void
vort_gpu_kernel_1(float *ctsurf, float *ctbot, 
				  float *cpvf, float *cjbar,
				  float *cadv, float *cten, 
				  float *ctot,
				  int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;


	if (i > 0 && i < im &&
		j > 0 && j < jm){
		
		ctot[j_off+i] = ctsurf[j_off+i]+ctbot[j_off+i]
					+cpvf[j_off+i]+cjbar[j_off+i]
					+cten[j_off+i]+cadv[j_off+i];
		//totx[j_off+i] = totx[j_off+i]*dum[j_off+i]*dum[j_off+i-1];
		//toty[j_off+i] = toty[j_off+i]*dvm[j_off+i]*dvm[j_1_off+i];
	}
}


__global__ void
vort_curl_gpu_kernel_0(float * __restrict__ ctsurf, 
		               float * __restrict__ ctbot,
					   float * __restrict__ celg, 
					   float * __restrict__ cjbar,
					   float * __restrict__ cadv, 
					   float * __restrict__ cpvf,
					   float * __restrict__ cten, 
					   const float * __restrict__ fx_ctsurf, 
					   const float * __restrict__ fy_ctsurf, 
					   const float * __restrict__ fx_ctbot, 
					   const float * __restrict__ fy_ctbot, 
					   const float * __restrict__ fx_celg, 
					   const float * __restrict__ fy_celg, 
					   const float * __restrict__ fx_cjbar, 
					   const float * __restrict__ fy_cjbar, 
					   const float * __restrict__ fx_cadv, 
					   const float * __restrict__ fy_cadv, 
					   const float * __restrict__ fx_cpvf, 
					   const float * __restrict__ fy_cpvf, 
					   const float * __restrict__ fx_cten, 
					   const float * __restrict__ fy_cten, 
				  	   const float * __restrict__ dx, 
					   const float * __restrict__ dy, 
				  	   const float * __restrict__ dum, 
					   const float * __restrict__ dvm, 
		               int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	float c, area;

	if (i > 0 && i < im &&
		j > 0 && j < jm){
		
		area = 0.25f*(dx[j_off+i]+dx[j_off+i-1]
					 +dx[j_1_off+i]+dx[j_1_off+i-1])
			  *0.25f*(dy[j_off+i]+dy[j_1_off+i]
					 +dy[j_off+i-1]+dy[j_1_off+i-1]);

		c = -fx_ctsurf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctsurf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctsurf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctsurf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctsurf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
				  *dvm[j_off+i]*dvm[j_off+i-1];

		ctsurf[j_off+i] = ctsurf[j_off+i]/area;

		c = -fx_ctbot[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctbot[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctbot[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctbot[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctbot[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		ctbot[j_off+i] = ctbot[j_off+i]/area;


		c = -fx_celg[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_celg[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_celg[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_celg[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		celg[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		celg[j_off+i] = celg[j_off+i]/area;


		c = -fx_cjbar[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cjbar[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cjbar[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cjbar[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cjbar[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		cjbar[j_off+i] = cjbar[j_off+i]/area;


		c = -fx_cadv[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cadv[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cadv[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cadv[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cadv[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cadv[j_off+i] = cadv[j_off+i]/area;

		c = -fx_cpvf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cpvf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cpvf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cpvf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cpvf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cpvf[j_off+i] = cpvf[j_off+i]/area;

		c = -fx_cten[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cten[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cten[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cten[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cten[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cten[j_off+i] = cten[j_off+i]/area;
	}
}

__global__ void
vort_curl_inner_gpu_kernel_0(float * __restrict__ ctsurf, 
		               float * __restrict__ ctbot,
					   float * __restrict__ celg, 
					   float * __restrict__ cjbar,
					   float * __restrict__ cadv, 
					   float * __restrict__ cpvf,
					   float * __restrict__ cten, 
					   const float * __restrict__ fx_ctsurf, 
					   const float * __restrict__ fy_ctsurf, 
					   const float * __restrict__ fx_ctbot, 
					   const float * __restrict__ fy_ctbot, 
					   const float * __restrict__ fx_celg, 
					   const float * __restrict__ fy_celg, 
					   const float * __restrict__ fx_cjbar, 
					   const float * __restrict__ fy_cjbar, 
					   const float * __restrict__ fx_cadv, 
					   const float * __restrict__ fy_cadv, 
					   const float * __restrict__ fx_cpvf, 
					   const float * __restrict__ fy_cpvf, 
					   const float * __restrict__ fx_cten, 
					   const float * __restrict__ fy_cten, 
				  	   const float * __restrict__ dx, 
					   const float * __restrict__ dy, 
				  	   const float * __restrict__ dum, 
					   const float * __restrict__ dvm, 
		               int jm, int im){

	int j = blockDim.y*blockIdx.y + threadIdx.y+1;
	int i = blockDim.x*blockIdx.x + threadIdx.x+1;

	float c, area;

//	if (blockIdx.x > 0 && blockIdx.x < gridDim.x-1 &&
//		blockIdx.y > 0 && blockIdx.y < gridDim.y-1){
	if (i > 32 && j > 32 && j < jm-33 && i < im-33){
		
		area = 0.25f*(dx[j_off+i]+dx[j_off+i-1]
					 +dx[j_1_off+i]+dx[j_1_off+i-1])
			  *0.25f*(dy[j_off+i]+dy[j_1_off+i]
					 +dy[j_off+i-1]+dy[j_1_off+i-1]);

		c = -fx_ctsurf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctsurf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctsurf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctsurf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctsurf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
				  *dvm[j_off+i]*dvm[j_off+i-1];

		ctsurf[j_off+i] = ctsurf[j_off+i]/area;

		c = -fx_ctbot[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctbot[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctbot[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctbot[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctbot[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		ctbot[j_off+i] = ctbot[j_off+i]/area;


		c = -fx_celg[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_celg[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_celg[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_celg[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		celg[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		celg[j_off+i] = celg[j_off+i]/area;


		c = -fx_cjbar[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cjbar[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cjbar[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cjbar[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cjbar[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		cjbar[j_off+i] = cjbar[j_off+i]/area;


		c = -fx_cadv[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cadv[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cadv[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cadv[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cadv[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cadv[j_off+i] = cadv[j_off+i]/area;

		c = -fx_cpvf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cpvf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cpvf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cpvf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cpvf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cpvf[j_off+i] = cpvf[j_off+i]/area;

		c = -fx_cten[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cten[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cten[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cten[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cten[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cten[j_off+i] = cten[j_off+i]/area;
	}
}

__global__ void
vort_curl_ew_gpu_kernel_0(
		               float * __restrict__ ctsurf, 
		               float * __restrict__ ctbot,
					   float * __restrict__ celg, 
					   float * __restrict__ cjbar,
					   float * __restrict__ cadv, 
					   float * __restrict__ cpvf,
					   float * __restrict__ cten, 
					   const float * __restrict__ fx_ctsurf, 
					   const float * __restrict__ fy_ctsurf, 
					   const float * __restrict__ fx_ctbot, 
					   const float * __restrict__ fy_ctbot, 
					   const float * __restrict__ fx_celg, 
					   const float * __restrict__ fy_celg, 
					   const float * __restrict__ fx_cjbar, 
					   const float * __restrict__ fy_cjbar, 
					   const float * __restrict__ fx_cadv, 
					   const float * __restrict__ fy_cadv, 
					   const float * __restrict__ fx_cpvf, 
					   const float * __restrict__ fy_cpvf, 
					   const float * __restrict__ fx_cten, 
					   const float * __restrict__ fy_cten, 
				  	   const float * __restrict__ dx, 
					   const float * __restrict__ dy, 
				  	   const float * __restrict__ dum, 
					   const float * __restrict__ dvm, 
		               int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	int i;

	float c, area;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;
	}else{
		i = im-2-threadIdx.x;	
	}
		
	if (j < jm-1){
		area = 0.25f*(dx[j_off+i]+dx[j_off+i-1]
					 +dx[j_1_off+i]+dx[j_1_off+i-1])
			  *0.25f*(dy[j_off+i]+dy[j_1_off+i]
					 +dy[j_off+i-1]+dy[j_1_off+i-1]);

		c = -fx_ctsurf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctsurf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctsurf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctsurf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctsurf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
				  *dvm[j_off+i]*dvm[j_off+i-1];

		ctsurf[j_off+i] = ctsurf[j_off+i]/area;

		c = -fx_ctbot[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctbot[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctbot[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctbot[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctbot[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		ctbot[j_off+i] = ctbot[j_off+i]/area;


		c = -fx_celg[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_celg[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_celg[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_celg[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		celg[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		celg[j_off+i] = celg[j_off+i]/area;


		c = -fx_cjbar[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cjbar[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cjbar[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cjbar[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cjbar[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		cjbar[j_off+i] = cjbar[j_off+i]/area;


		c = -fx_cadv[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cadv[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cadv[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cadv[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cadv[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cadv[j_off+i] = cadv[j_off+i]/area;

		c = -fx_cpvf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cpvf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cpvf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cpvf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cpvf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cpvf[j_off+i] = cpvf[j_off+i]/area;

		c = -fx_cten[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cten[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cten[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cten[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cten[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cten[j_off+i] = cten[j_off+i]/area;
	}

}

__global__ void
vort_curl_ew_bcond_gpu_kernel_0(float * __restrict__ ctsurf, 
		               float * __restrict__ ctbot,
					   float * __restrict__ celg, 
					   float * __restrict__ cjbar,
					   float * __restrict__ cadv, 
					   float * __restrict__ cpvf,
					   float * __restrict__ cten, 
					   const float * __restrict__ fx_ctsurf, 
					   const float * __restrict__ fy_ctsurf, 
					   const float * __restrict__ fx_ctbot, 
					   const float * __restrict__ fy_ctbot, 
					   const float * __restrict__ fx_celg, 
					   const float * __restrict__ fy_celg, 
					   const float * __restrict__ fx_cjbar, 
					   const float * __restrict__ fy_cjbar, 
					   const float * __restrict__ fx_cadv, 
					   const float * __restrict__ fy_cadv, 
					   const float * __restrict__ fx_cpvf, 
					   const float * __restrict__ fy_cpvf, 
					   const float * __restrict__ fx_cten, 
					   const float * __restrict__ fy_cten, 
				  	   const float * __restrict__ dx, 
					   const float * __restrict__ dy, 
				  	   const float * __restrict__ dum, 
					   const float * __restrict__ dvm, 
		               int n_west, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int i = im-1;

	float c, area;

	if (j < jm-1){
		area = 0.25f*(dx[j_off+i]+dx[j_off+i-1]
					 +dx[j_1_off+i]+dx[j_1_off+i-1])
			  *0.25f*(dy[j_off+i]+dy[j_1_off+i]
					 +dy[j_off+i-1]+dy[j_1_off+i-1]);

		c = -fx_ctsurf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctsurf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctsurf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctsurf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctsurf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
				  *dvm[j_off+i]*dvm[j_off+i-1];

		ctsurf[j_off+i] = ctsurf[j_off+i]/area;

		c = -fx_ctbot[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctbot[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctbot[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctbot[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctbot[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		ctbot[j_off+i] = ctbot[j_off+i]/area;


		c = -fx_celg[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_celg[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_celg[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_celg[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		celg[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		celg[j_off+i] = celg[j_off+i]/area;


		c = -fx_cjbar[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cjbar[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cjbar[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cjbar[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cjbar[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		cjbar[j_off+i] = cjbar[j_off+i]/area;


		c = -fx_cadv[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cadv[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cadv[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cadv[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cadv[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cadv[j_off+i] = cadv[j_off+i]/area;

		c = -fx_cpvf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cpvf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cpvf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cpvf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cpvf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cpvf[j_off+i] = cpvf[j_off+i]/area;

		c = -fx_cten[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cten[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cten[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cten[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cten[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cten[j_off+i] = cten[j_off+i]/area;
	}
	
	if (n_west == -1){
		ctsurf[j_off] = 0;
		ctbot[j_off] = 0;
		celg[j_off] = 0;
		cjbar[j_off] = 0;
		cadv[j_off] = 0;
		cpvf[j_off] = 0;
		cten[j_off] = 0;
	}

}

__global__ void
vort_curl_sn_gpu_kernel_0(float * __restrict__ ctsurf, 
		               float * __restrict__ ctbot,
					   float * __restrict__ celg, 
					   float * __restrict__ cjbar,
					   float * __restrict__ cadv, 
					   float * __restrict__ cpvf,
					   float * __restrict__ cten, 
					   const float * __restrict__ fx_ctsurf, 
					   const float * __restrict__ fy_ctsurf, 
					   const float * __restrict__ fx_ctbot, 
					   const float * __restrict__ fy_ctbot, 
					   const float * __restrict__ fx_celg, 
					   const float * __restrict__ fy_celg, 
					   const float * __restrict__ fx_cjbar, 
					   const float * __restrict__ fy_cjbar, 
					   const float * __restrict__ fx_cadv, 
					   const float * __restrict__ fy_cadv, 
					   const float * __restrict__ fx_cpvf, 
					   const float * __restrict__ fy_cpvf, 
					   const float * __restrict__ fx_cten, 
					   const float * __restrict__ fy_cten, 
				  	   const float * __restrict__ dx, 
					   const float * __restrict__ dy, 
				  	   const float * __restrict__ dum, 
					   const float * __restrict__ dvm, 
		               int jm, int im){

	int j;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

	float c, area;

	if (blockIdx.y < 8 ){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);	
	}
	
	if (i > 32 && i < im-33){
		area = 0.25f*(dx[j_off+i]+dx[j_off+i-1]
					 +dx[j_1_off+i]+dx[j_1_off+i-1])
			  *0.25f*(dy[j_off+i]+dy[j_1_off+i]
					 +dy[j_off+i-1]+dy[j_1_off+i-1]);

		c = -fx_ctsurf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctsurf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctsurf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctsurf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctsurf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
				  *dvm[j_off+i]*dvm[j_off+i-1];

		ctsurf[j_off+i] = ctsurf[j_off+i]/area;

		c = -fx_ctbot[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctbot[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctbot[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctbot[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctbot[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		ctbot[j_off+i] = ctbot[j_off+i]/area;


		c = -fx_celg[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_celg[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_celg[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_celg[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		celg[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		celg[j_off+i] = celg[j_off+i]/area;


		c = -fx_cjbar[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cjbar[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cjbar[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cjbar[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cjbar[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		cjbar[j_off+i] = cjbar[j_off+i]/area;


		c = -fx_cadv[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cadv[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cadv[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cadv[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cadv[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cadv[j_off+i] = cadv[j_off+i]/area;

		c = -fx_cpvf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cpvf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cpvf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cpvf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cpvf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cpvf[j_off+i] = cpvf[j_off+i]/area;

		c = -fx_cten[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cten[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cten[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cten[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cten[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cten[j_off+i] = cten[j_off+i]/area;
	}
}

__global__ void
vort_curl_sn_bcond_gpu_kernel_0(float * __restrict__ ctsurf, 
		               float * __restrict__ ctbot,
					   float * __restrict__ celg, 
					   float * __restrict__ cjbar,
					   float * __restrict__ cadv, 
					   float * __restrict__ cpvf,
					   float * __restrict__ cten, 
					   const float * __restrict__ fx_ctsurf, 
					   const float * __restrict__ fy_ctsurf, 
					   const float * __restrict__ fx_ctbot, 
					   const float * __restrict__ fy_ctbot, 
					   const float * __restrict__ fx_celg, 
					   const float * __restrict__ fy_celg, 
					   const float * __restrict__ fx_cjbar, 
					   const float * __restrict__ fy_cjbar, 
					   const float * __restrict__ fx_cadv, 
					   const float * __restrict__ fy_cadv, 
					   const float * __restrict__ fx_cpvf, 
					   const float * __restrict__ fy_cpvf, 
					   const float * __restrict__ fx_cten, 
					   const float * __restrict__ fy_cten, 
				  	   const float * __restrict__ dx, 
					   const float * __restrict__ dy, 
				  	   const float * __restrict__ dum, 
					   const float * __restrict__ dvm, 
		               int n_south, int jm, int im){

	const int j = jm-1;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

	float c, area;

	if (i < im-1){
		area = 0.25f*(dx[j_off+i]+dx[j_off+i-1]
					 +dx[j_1_off+i]+dx[j_1_off+i-1])
			  *0.25f*(dy[j_off+i]+dy[j_1_off+i]
					 +dy[j_off+i-1]+dy[j_1_off+i-1]);

		c = -fx_ctsurf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctsurf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctsurf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctsurf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctsurf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
				  *dvm[j_off+i]*dvm[j_off+i-1];

		ctsurf[j_off+i] = ctsurf[j_off+i]/area;

		c = -fx_ctbot[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_ctbot[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_ctbot[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_ctbot[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		ctbot[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		ctbot[j_off+i] = ctbot[j_off+i]/area;


		c = -fx_celg[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_celg[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_celg[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_celg[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		celg[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		celg[j_off+i] = celg[j_off+i]/area;


		c = -fx_cjbar[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cjbar[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cjbar[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cjbar[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cjbar[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						  *dvm[j_off+i]*dvm[j_off+i-1];

		cjbar[j_off+i] = cjbar[j_off+i]/area;


		c = -fx_cadv[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cadv[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cadv[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cadv[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cadv[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cadv[j_off+i] = cadv[j_off+i]/area;

		c = -fx_cpvf[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cpvf[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cpvf[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cpvf[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cpvf[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cpvf[j_off+i] = cpvf[j_off+i]/area;

		c = -fx_cten[j_off+i]*(dx[j_off+i]+dx[j_off+i-1])
			+fx_cten[j_1_off+i]*(dx[j_1_off+i]+dx[j_1_off+i-1])
			+fy_cten[j_off+i]*(dy[j_off+i]+dy[j_1_off+i])
			-fy_cten[j_off+i-1]*(dy[j_off+i-1]+dy[j_1_off+i-1]);

		cten[j_off+i] = c*dum[j_off+i]*dum[j_1_off+i]
						 *dvm[j_off+i]*dvm[j_off+i-1];

		cten[j_off+i] = cten[j_off+i]/area;
	}

	if (n_south == -1){
		ctsurf[i] = 0;
		ctbot[i] = 0;
		celg[i] = 0;
		cjbar[i] = 0;
		cadv[i] = 0;
		cpvf[i] = 0;
		cten[i] = 0;
	}
}

void vort_gpu(){

#ifndef TIME_DISABLE
	struct timeval start_vort,
				   end_vort;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_vort);
#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
					  (j_size+block_j_2D-1)/block_j_2D);

	dim3 threadPerBlock_ew(32, 4);
	dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	dim3 threadPerBlock_ew_bcond(1, 128);
	dim3 blockPerGrid_ew_bcond(1, (j_size-2+127)/128);

	dim3 threadPerBlock_sn(32, 4);
	dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	dim3 threadPerBlock_sn_bcond(128, 1);
	dim3 blockPerGrid_sn_bcond((i_size-2+127)/128, 1);

	int iid = 1;
	float *d_fx_ctsurf = d_2d_tmp0;
	float *d_fx_ctsurf_east = d_2d_tmp0_east;
	float *d_fx_ctsurf_west = d_2d_tmp0_west;
	float *d_fx_ctbot = d_2d_tmp1;
	float *d_fx_ctbot_east = d_2d_tmp1_east;
	float *d_fx_ctbot_west = d_2d_tmp1_west;
	float *d_fx_celg = d_2d_tmp2;
	float *d_fx_celg_east = d_2d_tmp2_east;
	float *d_fx_celg_west = d_2d_tmp2_west;
	float *d_fx_cjbar = d_2d_tmp3;
	float *d_fx_cjbar_east = d_2d_tmp3_east;
	float *d_fx_cjbar_west = d_2d_tmp3_west;
	float *d_fx_cadv = d_2d_tmp4;
	float *d_fx_cadv_east = d_2d_tmp4_east;
	float *d_fx_cadv_west = d_2d_tmp4_west;
	float *d_fx_cpvf = d_2d_tmp5;
	float *d_fx_cpvf_east = d_2d_tmp5_east;
	float *d_fx_cpvf_west = d_2d_tmp5_west;
	float *d_fx_cten = d_2d_tmp6;
	float *d_fx_cten_east = d_2d_tmp6_east;
	float *d_fx_cten_west = d_2d_tmp6_west;

	float *d_fy_ctsurf = d_2d_tmp7;
	float *d_fy_ctsurf_east = d_2d_tmp7_east;
	float *d_fy_ctsurf_west = d_2d_tmp7_west;
	float *d_fy_ctbot = d_2d_tmp8;
	float *d_fy_ctbot_east = d_2d_tmp8_east;
	float *d_fy_ctbot_west = d_2d_tmp8_west;
	float *d_fy_celg = d_2d_tmp9;
	float *d_fy_celg_east = d_2d_tmp9_east;
	float *d_fy_celg_west = d_2d_tmp9_west;
	float *d_fy_cjbar = d_2d_tmp10;
	float *d_fy_cjbar_east = d_2d_tmp10_east;
	float *d_fy_cjbar_west = d_2d_tmp10_west;
	float *d_fy_cadv = d_2d_tmp11;
	float *d_fy_cadv_east = d_2d_tmp11_east;
	float *d_fy_cadv_west = d_2d_tmp11_west;
	float *d_fy_cpvf = d_2d_tmp12;
	float *d_fy_cpvf_east = d_2d_tmp12_east;
	float *d_fy_cpvf_west = d_2d_tmp12_west;
	float *d_fy_cten = d_2d_tmp13;
	float *d_fy_cten_east = d_2d_tmp13_east;
	float *d_fy_cten_west = d_2d_tmp13_west;

	//vort_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
	//		d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
	//		d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
	//		d_fx_cten, d_fy_cten,
	//		d_ctsurf, d_ctbot, d_cpvf, d_cjbar, 
	//		d_cadv, d_cten, d_celg,
	//		d_totx, d_toty, d_d, 
	//		d_elb, d_el, d_elf,
	//		d_drx2d, d_dry2d,
	//		d_adx2d, d_ady2d, d_advua, d_advva,
	//		d_uab, d_vab, d_ua, d_va, d_uaf, d_vaf,
	//		d_wusurf, d_wvsurf, d_wubot, d_wvbot,
	//		d_dum, d_dvm, d_aru, d_arv,
	//		d_dx, d_dy, d_cor, d_h,
	//		grav, dte, iid, jm, im);



	////exchange2d_mpi_gpu(d_fx_ctsurf, im, jm);
	////exchange2d_mpi_gpu(d_fx_ctbot, im, jm);
	////exchange2d_mpi_gpu(d_fx_celg, im, jm);
	////exchange2d_mpi_gpu(d_fx_cjbar, im, jm);
	////exchange2d_mpi_gpu(d_fx_cadv, im, jm);
	////exchange2d_mpi_gpu(d_fx_cpvf, im, jm);
	////exchange2d_mpi_gpu(d_fx_cten, im, jm);
	////exchange2d_mpi_gpu(d_fy_ctsurf, im, jm);
	////exchange2d_mpi_gpu(d_fy_ctbot, im, jm);
	////exchange2d_mpi_gpu(d_fy_celg,  im, jm);
	////exchange2d_mpi_gpu(d_fy_cjbar, im, jm);
	////exchange2d_mpi_gpu(d_fy_cadv,  im, jm);
	////exchange2d_mpi_gpu(d_fy_cpvf,  im, jm);
	////exchange2d_mpi_gpu(d_fy_cten,  im, jm);
	////exchange2d_mpi_gpu(d_totx, im, jm);
	////exchange2d_mpi_gpu(d_toty, im, jm);

	//exchange2d_cuda_aware_mpi(d_fx_ctsurf, im, jm);
	//exchange2d_cuda_aware_mpi(d_fx_ctbot, im, jm);
	//exchange2d_cuda_aware_mpi(d_fx_celg, im, jm);
	//exchange2d_cuda_aware_mpi(d_fx_cjbar, im, jm);
	//exchange2d_cuda_aware_mpi(d_fx_cadv, im, jm);
	//exchange2d_cuda_aware_mpi(d_fx_cpvf, im, jm);
	//exchange2d_cuda_aware_mpi(d_fx_cten, im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_ctsurf, im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_ctbot, im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_celg,  im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_cjbar, im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_cadv,  im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_cpvf,  im, jm);
	//exchange2d_cuda_aware_mpi(d_fy_cten,  im, jm);
	//exchange2d_cuda_aware_mpi(d_totx, im, jm);
	//exchange2d_cuda_aware_mpi(d_toty, im, jm);


	vort_ew_gpu_kernel_0<<<blockPerGrid_ew, threadPerBlock_ew,
						   0, stream[1]>>>(
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_ctsurf, d_ctbot, d_cpvf, d_cjbar, 
			d_cadv, d_cten, d_celg,
			d_totx, d_toty, d_d, 
			d_elb, d_el, d_elf,
			d_drx2d, d_dry2d,
			d_adx2d, d_ady2d, d_advua, d_advva,
			d_uab, d_vab, d_ua, d_va, d_uaf, d_vaf,
			d_wusurf, d_wvsurf, d_wubot, d_wvbot,
			d_dum, d_dvm, d_aru, d_arv,
			d_dx, d_dy, d_cor, d_h,
			grav, dte, iid, jm, im);

	vort_sn_gpu_kernel_0<<<blockPerGrid_sn, threadPerBlock_sn,
						   0, stream[2]>>>(
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_ctsurf, d_ctbot, d_cpvf, d_cjbar, 
			d_cadv, d_cten, d_celg,
			d_totx, d_toty, d_d, 
			d_elb, d_el, d_elf,
			d_drx2d, d_dry2d,
			d_adx2d, d_ady2d, d_advua, d_advva,
			d_uab, d_vab, d_ua, d_va, d_uaf, d_vaf,
			d_wusurf, d_wvsurf, d_wubot, d_wvbot,
			d_dum, d_dvm, d_aru, d_arv,
			d_dx, d_dy, d_cor, d_h,
			grav, dte, iid, jm, im);



	vort_inner_gpu_kernel_0<<<blockPerGrid, threadPerBlock,
							  0, stream[0]>>>(
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_ctsurf, d_ctbot, d_cpvf, d_cjbar, 
			d_cadv, d_cten, d_celg,
			d_totx, d_toty, d_d, 
			d_elb, d_el, d_elf,
			d_drx2d, d_dry2d,
			d_adx2d, d_ady2d, d_advua, d_advva,
			d_uab, d_vab, d_ua, d_va, d_uaf, d_vaf,
			d_wusurf, d_wvsurf, d_wubot, d_wvbot,
			d_dum, d_dvm, d_aru, d_arv,
			d_dx, d_dy, d_cor, d_h,
			grav, dte, iid, jm, im);

	vort_ew_bcond_gpu_kernel_0<<<blockPerGrid_ew_bcond, 
								 threadPerBlock_ew_bcond,
						   0, stream[3]>>>(
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_ctsurf, d_ctbot, d_cpvf, d_cjbar, 
			d_cadv, d_cten, d_celg,
			d_totx, d_toty, d_d, 
			d_elb, d_el, d_elf,
			d_drx2d, d_dry2d,
			d_adx2d, d_ady2d, d_advua, d_advva,
			d_uab, d_vab, d_ua, d_va, d_uaf, d_vaf,
			d_wusurf, d_wvsurf, d_wubot, d_wvbot,
			d_dum, d_dvm, d_aru, d_arv,
			d_dx, d_dy, d_cor, d_h,
			grav, dte, iid, n_west, jm, im);

	vort_sn_bcond_gpu_kernel_0<<<blockPerGrid_sn_bcond, 
								 threadPerBlock_sn_bcond,
						   0, stream[4]>>>(
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_ctsurf, d_ctbot, d_cpvf, d_cjbar, 
			d_cadv, d_cten, d_celg,
			d_totx, d_toty, d_d, 
			d_elb, d_el, d_elf,
			d_drx2d, d_dry2d,
			d_adx2d, d_ady2d, d_advua, d_advva,
			d_uab, d_vab, d_ua, d_va, d_uaf, d_vaf,
			d_wusurf, d_wvsurf, d_wubot, d_wvbot,
			d_dum, d_dvm, d_aru, d_arv,
			d_dx, d_dy, d_cor, d_h,
			grav, dte, iid, n_south, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

	//exchange2d_cuda_ipc(d_fx_ctsurf, d_fx_ctsurf_east, d_fx_ctsurf_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_fx_ctbot, d_fx_ctbot_east, d_fx_ctbot_west,
	//					stream[2], im, jm);
	//exchange2d_cuda_ipc(d_fx_celg, d_fx_celg_east, d_fx_celg_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_fx_cjbar, d_fx_cjbar_east, d_fx_cjbar_west,
	//					stream[2], im, jm);
	//exchange2d_cuda_ipc(d_fx_cadv, d_fx_cadv_east, d_fx_cadv_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_fx_cpvf, d_fx_cpvf_east, d_fx_cpvf_west,
	//					stream[2], im, jm);
	//exchange2d_cuda_ipc(d_fx_cten, d_fx_cten_east, d_fx_cten_west,
	//					stream[1], im, jm);

	//exchange2d_cuda_ipc(d_fy_ctsurf, d_fy_ctsurf_east, d_fy_ctsurf_west,
	//					stream[2], im, jm);
	//exchange2d_cuda_ipc(d_fy_ctbot, d_fy_ctbot_east, d_fy_ctbot_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_fy_celg, d_fy_celg_east, d_fy_celg_west,
	//					stream[2], im, jm);
	//exchange2d_cuda_ipc(d_fy_cjbar, d_fy_cjbar_east, d_fy_cjbar_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_fy_cadv, d_fy_cadv_east, d_fy_cadv_west,
	//					stream[2], im, jm);
	//exchange2d_cuda_ipc(d_fy_cpvf, d_fy_cpvf_east, d_fy_cpvf_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_fy_cten, d_fy_cten_east, d_fy_cten_west,
	//					stream[2], im, jm);

	//exchange2d_cuda_ipc(d_totx, d_totx_east, d_totx_west,
	//					stream[1], im, jm);
	//exchange2d_cuda_ipc(d_toty, d_toty_east, d_toty_west,
	//					stream[2], im, jm);

	//MPI_Barrier(pom_comm);
	//exchange2d_cudaPeerAsync(d_fx_ctsurf, 
	//					d_fx_ctsurf_east, d_fx_ctsurf_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_ctbot, 
	//					d_fx_ctbot_east, d_fx_ctbot_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_celg, 
	//					d_fx_celg_east, d_fx_celg_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_cjbar, 
	//					d_fx_cjbar_east, d_fx_cjbar_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_cadv, 
	//					d_fx_cadv_east, d_fx_cadv_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_cpvf, 
	//					d_fx_cpvf_east, d_fx_cpvf_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_cten, 
	//					d_fx_cten_east, d_fx_cten_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fx_cten, 
	//					d_fx_cten_east, d_fx_cten_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);

	//exchange2d_cudaPeerAsync(d_fy_ctsurf, 
	//					d_fy_ctsurf_east, d_fy_ctsurf_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_ctbot, 
	//					d_fy_ctbot_east, d_fy_ctbot_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_celg, 
	//					d_fy_celg_east, d_fy_celg_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_cjbar, 
	//					d_fy_cjbar_east, d_fy_cjbar_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_cadv, 
	//					d_fy_cadv_east, d_fy_cadv_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_cpvf, 
	//					d_fy_cpvf_east, d_fy_cpvf_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_cten, 
	//					d_fy_cten_east, d_fy_cten_west,
	//					NULL, NULL,
	//				    stream[2], im, jm);
	//exchange2d_cudaPeerAsync(d_fy_cten, 
	//					d_fy_cten_east, d_fy_cten_west,
	//					NULL, NULL,
	//				    stream[1], im, jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//checkCudaErrors(cudaStreamSynchronize(stream[2]));
	//MPI_Barrier(pom_comm);

	//MPI_Barrier(pom_comm);
	//send_east_cuda_ipc(d_fx_ctsurf, d_fx_ctsurf_east, 
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_fx_ctbot, d_fx_ctbot_east, 
	//				   stream[2], im, jm);
	//send_east_cuda_ipc(d_fx_celg, d_fx_celg_east,
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_fx_cjbar, d_fx_cjbar_east, 
	//				   stream[2], im, jm);
	//send_east_cuda_ipc(d_fx_cadv, d_fx_cadv_east,
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_fx_cpvf, d_fx_cpvf_east,
	//				   stream[2], im, jm);
	//send_east_cuda_ipc(d_fx_cten, d_fx_cten_east,
	//				   stream[1], im, jm);

	//send_east_cuda_ipc(d_fy_ctsurf, d_fy_ctsurf_east, 
	//				   stream[2], im, jm);
	//send_east_cuda_ipc(d_fy_ctbot, d_fy_ctbot_east, 
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_fy_celg, d_fy_celg_east, 
	//				   stream[2], im, jm);
	//send_east_cuda_ipc(d_fy_cjbar, d_fy_cjbar_east,
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_fy_cadv, d_fy_cadv_east,
	//				   stream[2], im, jm);
	//send_east_cuda_ipc(d_fy_cpvf, d_fy_cpvf_east,
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_fy_cten, d_fy_cten_east,
	//				   stream[2], im, jm);

	//send_east_cuda_ipc(d_totx, d_totx_east,
	//				   stream[1], im, jm);
	//send_east_cuda_ipc(d_toty, d_toty_east,
	//				   stream[2], im, jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//checkCudaErrors(cudaStreamSynchronize(stream[2]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[3]));
	checkCudaErrors(cudaStreamSynchronize(stream[4]));
	checkCudaErrors(cudaStreamSynchronize(stream[0]));

	//vort_curl_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		d_ctsurf, d_ctbot, d_celg, d_cjbar, 
	//		d_cadv, d_cpvf, d_cten,
	//		d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
	//		d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
	//		d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
	//		d_fx_cten, d_fy_cten,
	//		d_dx, d_dy, d_dum, d_dvm, 
	//		jm, im);

	//exchange2d_cuda_aware_mpi(d_ctsurf, im, jm);
	//exchange2d_cuda_aware_mpi(d_ctbot, im, jm);
	//exchange2d_cuda_aware_mpi(d_celg, im, jm);
	//exchange2d_cuda_aware_mpi(d_cjbar, im, jm);
	//exchange2d_cuda_aware_mpi(d_cadv, im, jm);
	//exchange2d_cuda_aware_mpi(d_cpvf, im, jm);
	//exchange2d_cuda_aware_mpi(d_cten, im, jm);

	vort_curl_ew_gpu_kernel_0<<<blockPerGrid_ew, threadPerBlock_ew, 
								   0, stream[1]>>>(
			d_ctsurf, d_ctbot, d_celg, d_cjbar, 
			d_cadv, d_cpvf, d_cten,
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_dx, d_dy, d_dum, d_dvm, 
			jm, im);

	vort_curl_sn_gpu_kernel_0<<<blockPerGrid_sn, threadPerBlock_sn, 
								   0, stream[2]>>>(
			d_ctsurf, d_ctbot, d_celg, d_cjbar, 
			d_cadv, d_cpvf, d_cten,
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_dx, d_dy, d_dum, d_dvm, 
			jm, im);


	vort_curl_inner_gpu_kernel_0<<<blockPerGrid, threadPerBlock, 
								   0, stream[0]>>>(
			d_ctsurf, d_ctbot, d_celg, d_cjbar, 
			d_cadv, d_cpvf, d_cten,
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_dx, d_dy, d_dum, d_dvm, 
			jm, im);

	vort_curl_ew_bcond_gpu_kernel_0<<<blockPerGrid_ew_bcond, 
									  threadPerBlock_ew_bcond, 
								   0, stream[3]>>>(
			d_ctsurf, d_ctbot, d_celg, d_cjbar, 
			d_cadv, d_cpvf, d_cten,
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_dx, d_dy, d_dum, d_dvm, 
			n_west, jm, im);

	vort_curl_sn_bcond_gpu_kernel_0<<<blockPerGrid_sn_bcond, 
									  threadPerBlock_sn_bcond, 
								   0, stream[4]>>>(
			d_ctsurf, d_ctbot, d_celg, d_cjbar, 
			d_cadv, d_cpvf, d_cten,
			d_fx_ctsurf, d_fy_ctsurf, d_fx_ctbot, d_fy_ctbot,
			d_fx_celg, d_fy_celg, d_fx_cjbar, d_fy_cjbar,
			d_fx_cadv, d_fy_cadv, d_fx_cpvf, d_fy_cpvf,
			d_fx_cten, d_fy_cten,
			d_dx, d_dy, d_dum, d_dvm, 
			n_south, jm, im);


	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

	//MPI_Barrier(pom_comm);
	//exchange2d_cudaPeerAsync(d_ctsurf, 
	//					d_ctsurf_east, d_ctsurf_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_ctbot, 
	//					d_ctbot_east, d_ctbot_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_celg, 
	//					d_celg_east, d_celg_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_cjbar, 
	//					d_cjbar_east, d_cjbar_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_cadv, 
	//					d_cadv_east, d_cadv_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_cpvf, 
	//					d_cpvf_east, d_cpvf_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//exchange2d_cudaPeerAsync(d_cten, 
	//					d_cten_east, d_cten_west,
	//					NULL, NULL,
	//					stream[1], im, jm);
	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);

	//MPI_Barrier(pom_comm);
	//send_east_cuda_ipc(d_ctsurf, d_ctsurf_east, 
	//					stream[1], im, jm);
	//send_east_cuda_ipc(d_ctbot, d_ctbot_east,
	//				    stream[2], im, jm);
	//send_east_cuda_ipc(d_celg, d_celg_east,
	//					stream[1], im, jm);
	//send_east_cuda_ipc(d_cjbar, d_cjbar_east,
	//					stream[2], im, jm);
	//send_east_cuda_ipc(d_cadv, d_cadv_east,
	//				    stream[1], im, jm);
	//send_east_cuda_ipc(d_cpvf, d_cpvf_east,
	//					stream[2], im, jm);
	//send_east_cuda_ipc(d_cten, d_cten_east, 
	//					stream[1], im, jm);
	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//checkCudaErrors(cudaStreamSynchronize(stream[2]));
	//MPI_Barrier(pom_comm);
	checkCudaErrors(cudaStreamSynchronize(stream[3]));
	checkCudaErrors(cudaStreamSynchronize(stream[4]));
	checkCudaErrors(cudaStreamSynchronize(stream[0]));

	vort_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_ctsurf, d_ctbot, d_cpvf, d_cjbar,
			d_cadv, d_cten, d_ctot, 
			jm, im);

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_vort);
		vort_time += time_consumed(&start_vort, 
								   &end_vort);
#endif

}

//void vort_gpu(){
//
//#ifndef TIME_DISABLE
//	struct timeval start_vort,
//				   end_vort;
//
//	checkCudaErrors(cudaDeviceSynchronize());
//	timer_now(&start_vort);
//#endif
//
//	if (iint == 2)
//		printf("vort_gpu feature is not supported now!\n\n");
//
//#ifndef TIME_DISABLE
//		checkCudaErrors(cudaDeviceSynchronize());
//		timer_now(&end_vort);
//		vort_time += time_consumed(&start_vort, 
//								   &end_vort);
//#endif
//	//exit(1);
//}


__global__ void
advq_fusion_gpu_kernel_0(
				  float * __restrict__ xflux_u, 
				  float * __restrict__ yflux_u, 
				  float * __restrict__ xflux_v, 
				  float * __restrict__ yflux_v, 
				  const float * __restrict__ qu, 
				  const float * __restrict__ qub,
				  const float * __restrict__ qv, 
				  const float * __restrict__ qvb,
				  const float * __restrict__ u, 
				  const float * __restrict__ v,
				  const float * __restrict__ aam, 
				  const float * __restrict__ dt, 
				  const float * __restrict__ dum, 
				  const float * __restrict__ dvm,
				  const float * __restrict__ h, 
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy,
				  int kb, int jm, int im){

	//modify -xflux, yflux
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;



	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 1; k < kbm1; k++){
			float xflux_u_tmp,  yflux_u_tmp;
			float xflux_v_tmp,  yflux_v_tmp;
			xflux_u_tmp = 0.125f*(qu[k_off+j_off+i]
								  +qu[k_off+j_off+(i-1)])
								 *(dt[j_off+i]
								  +dt[j_off+(i-1)])
								 *(u[k_off+j_off+i]
								  +u[k_1_off+j_off+i]);	

			xflux_v_tmp = 0.125f*(qv[k_off+j_off+i]
								  +qv[k_off+j_off+(i-1)])
								 *(dt[j_off+i]
								  +dt[j_off+(i-1)])
								 *(u[k_off+j_off+i]
								  +u[k_1_off+j_off+i]);	

			///////////////////////////////////////
			yflux_u_tmp = 0.125f*(qu[k_off+j_off+i]
								  +qu[k_off+j_1_off+i])
								 *(dt[j_off+i]
								  +dt[j_1_off+i])
								 *(v[k_off+j_off+i]
								  +v[k_1_off+j_off+i]);

			yflux_v_tmp = 0.125f*(qv[k_off+j_off+i]
								  +qv[k_off+j_1_off+i])
								 *(dt[j_off+i]
								  +dt[j_1_off+i])
								 *(v[k_off+j_off+i]
								  +v[k_1_off+j_off+i]);

			///////////////////////////////////////

			xflux_u_tmp = xflux_u_tmp
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_off+(i-1)]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_off+(i-1)])
								*(h[j_off+i]+h[j_off+(i-1)])
								*(qub[k_off+j_off+i]
								 -qub[k_off+j_off+(i-1)])
								*dum[j_off+i]
								/(dx[j_off+i]+dx[j_off+(i-1)]);	

			xflux_v_tmp = xflux_v_tmp
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_off+(i-1)]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_off+(i-1)])
								*(h[j_off+i]+h[j_off+(i-1)])
								*(qvb[k_off+j_off+i]
								 -qvb[k_off+j_off+(i-1)])
								*dum[j_off+i]
								/(dx[j_off+i]+dx[j_off+(i-1)]);	

			///////////////////////////////////////

			yflux_u_tmp = yflux_u_tmp
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_1_off+i]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_1_off+i])
								*(h[j_off+i]+h[j_1_off+i])
								*(qub[k_off+j_off+i]
								 -qub[k_off+j_1_off+i])
								*dvm[j_off+i]
								/(dy[j_off+i]+dy[j_1_off+i]);

			yflux_v_tmp = yflux_v_tmp
						  -0.25f*(aam[k_off+j_off+i]
								 +aam[k_off+j_1_off+i]
								 +aam[k_1_off+j_off+i]
								 +aam[k_1_off+j_1_off+i])
								*(h[j_off+i]+h[j_1_off+i])
								*(qvb[k_off+j_off+i]
								 -qvb[k_off+j_1_off+i])
								*dvm[j_off+i]
								/(dy[j_off+i]+dy[j_1_off+i]);

			///////////////////////////////////////

			xflux_u[k_off+j_off+i] = 0.5f*(dy[j_off+i]
										+dy[j_off+(i-1)])
									   *xflux_u_tmp;

			xflux_v[k_off+j_off+i] = 0.5f*(dy[j_off+i]
										+dy[j_off+(i-1)])
									   *xflux_v_tmp;

			///////////////////////////////////////

			yflux_u[k_off+j_off+i] = 0.5f*(dx[j_off+i]
										+dx[j_1_off+i])
									   *yflux_u_tmp;

			yflux_v[k_off+j_off+i] = 0.5f*(dx[j_off+i]
										+dx[j_1_off+i])
									   *yflux_v_tmp;

			///////////////////////////////////////
		}
	}


}

__global__ void
advq_fusion_gpu_kernel_1(
				  float * __restrict__ quf, 
				  const float * __restrict__ qub, 
				  const float * __restrict__ qu,
				  float * __restrict__ qvf, 
				  const float * __restrict__ qvb, 
				  const float * __restrict__ qv,
				  const float * __restrict__ w, 
				  const float * __restrict__ xflux_u, 
				  const float * __restrict__ yflux_u,
				  const float * __restrict__ xflux_v, 
				  const float * __restrict__ yflux_v,
				  const float * __restrict__ etb, 
				  const float * __restrict__ etf, 
				  const float * __restrict__ art, 
				  const float * __restrict__ dz, 
				  const float * __restrict__ h, 
				  float dti2, int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;


	if (j > 0 && j < jmm1 && i > 0 && i < imm1){
		for (k = 1; k < kbm1; k++){
			float tmp_u, tmp_v;
			tmp_u = (w[k_1_off+j_off+i]
						*qu[k_1_off+j_off+i]
					-w[k_A1_off+j_off+i]
						*qu[k_A1_off+j_off+i])
				   *art[j_off+i]/(dz[k]+dz[k-1])
				  +xflux_u[k_off+j_off+(i+1)]
				  -xflux_u[k_off+j_off+i]
				  +yflux_u[k_off+j_A1_off+i]
				  -yflux_u[k_off+j_off+i];	

			tmp_v = (w[k_1_off+j_off+i]
						*qv[k_1_off+j_off+i]
					-w[k_A1_off+j_off+i]
						*qv[k_A1_off+j_off+i])
				   *art[j_off+i]/(dz[k]+dz[k-1])
				  +xflux_v[k_off+j_off+(i+1)]
				  -xflux_v[k_off+j_off+i]
				  +yflux_v[k_off+j_A1_off+i]
				  -yflux_v[k_off+j_off+i];	

			///////////////////////////////////

			quf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i])
									*art[j_off+i]*qub[k_off+j_off+i]
								  -dti2*tmp_u)
								/((h[j_off+i]+etf[j_off+i])
									*art[j_off+i]);

			qvf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i])
									*art[j_off+i]*qvb[k_off+j_off+i]
								  -dti2*tmp_v)
								/((h[j_off+i]+etf[j_off+i])
									*art[j_off+i]);
		}
	}
}

/*
void advq_gpu(float qb[][j_size][i_size], float q[][j_size][i_size],
			  float qf[][j_size][i_size], float u[][j_size][i_size],
			  float dt[][i_size], float v[][j_size][i_size],
			  float aam[][j_size][i_size], float w[][j_size][i_size],
			  float etb[][i_size], float etf[][i_size]){
*/

/*
void advq_gpu(float *d_qb, float *d_q, float *d_qf, 
			  float *d_u, float *d_v, float *d_w,
			  float *d_etb, float *d_etf,
			  float *d_aam, float *d_dt){
*/
void advq_fusion_gpu(float *d_qub, float *d_qu, float *d_quf, 
					 float *d_qvb, float *d_qv, float *d_qvf){

#ifndef TIME_DISABLE
	struct timeval start_fusion_advq,
				   end_fusion_advq;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_fusion_advq);
#endif

	//modify:
	//     -qf
	//int i,j,k;
	/*
	float *d_qb = d_qb_advq;
	float *d_q = d_q_advq;
	float *d_qf = d_qf_advq;
	*/

	float *d_xflux_u = d_3d_tmp0;
	float *d_yflux_u = d_3d_tmp1;
	float *d_xflux_v = d_3d_tmp2;
	float *d_yflux_v = d_3d_tmp3;

    //float xflux[k_size][j_size][i_size];
	//float yflux[k_size][j_size][i_size];

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((im+block_i_2D-1)/block_i_2D, 
					  (jm+block_j_2D-1)/block_j_2D);
	

	advq_fusion_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xflux_u, d_yflux_u, d_xflux_v, d_yflux_v,
			d_qu, d_qub, d_qv, d_qvb,
			d_u, d_v, d_aam, d_dt, 
			d_dum, d_dvm, d_h, d_dx, d_dy, kb, jm, im);

    //exchange3d_mpi_gpu(d_xflux,im,jm,kbm1);
    //exchange3d_mpi_gpu(d_yflux,im,jm,kbm1);

//! do vertical advection, add flux terms, then step forward in time
	//modify -qf

	advq_fusion_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
			d_quf, d_qub, d_qu, d_qvf, d_qvb, d_qv, d_w, 
			d_xflux_u, d_yflux_u, d_xflux_v, d_yflux_v, d_etb, d_etf, 
			d_art, d_dz, d_h, dti2, kb, jm, im);
	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_fusion_advq);
		advq_fusion_time += time_consumed(&start_fusion_advq, 
								   &end_fusion_advq);
#endif

	return;

}

__global__ void
proft_fusion_gpu_kernel_0(float * __restrict__ kh, 
				   float * __restrict__ etf, 
				   float * __restrict__ swrad,
				   float * __restrict__ wfsurf_u, 
				   float * __restrict__ f_u, 
				   const float * __restrict__ fsurf_u,
				   float * __restrict__ wfsurf_v, 
				   float * __restrict__ f_v, 
				   const float * __restrict__ fsurf_v,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz, 
				   const float * __restrict__ z,
				   float dti2, float umol, int ntp, 
				   int nbc_u, int nbc_v,
				   int kb, int jm, int im){

	//		+ f
	int k, ki;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};

	float dh;
	//float a[k_size], c[k_size];
	//float ee[k_size], gg[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];
	float rad_u[k_size], rad_v[k_size];

	if (j < jm && i < im){
		dh = h[j_off+i]+etf[j_off+i];	

		//for (k = 1; k < kbm1; k++){
		//	
		//	a[k-1] = -dti2*(kh[k_off+j_off+i]+umol)
		//				  /(dz[k-1]*dzz[k-1]*dh*dh);
		//	
		//	c[k] = -dti2*(kh[k_off+j_off+i]+umol)
		//			    /(dz[k]*dzz[k-1]*dh*dh);
		//	
		//}

		for (k = 0; k < kb; k++){
			rad_u[k] = 0;	
			rad_v[k] = 0;	
		}
	
		if (nbc_u == 2 || nbc_u == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_u[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		if (nbc_v == 2 || nbc_v == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_v[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		float tmp_a0 = -dti2*(kh[jm*im+j_off+i]+umol)
					        /(dz[0]*dzz[0]*dh*dh);

		//if (nbc == 1){
		//	//ee[0] = a[0]/(a[0]-1.0f);
		//	//gg[0] = -dti2*wfsurf[j_off+i]/(-dz[0]*dh)
		//	//			  -f[j_off+i];
		//	//gg[0] = gg[0]/(a[0]-1.0f);

		//	//ee[0] = tmp_a0/(tmp_a0-1.0f);
		//	//gg[0] = -dti2*wfsurf[j_off+i]/(-dz[0]*dh)
		//	//			  -f[j_off+i];
		//	//gg[0] = gg[0]/(tmp_a0-1.0f);

		//	ee_u[0] = tmp_a0/(tmp_a0-1.0f);
		//	ee_v[0] = ee_u[0];

		//	gg_u[0] = -dti2*wfsurf_u[j_off+i]/(-dz[0]*dh)
		//				  -f_u[j_off+i];
		//	gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		//	gg_v[0] = -dti2*wfsurf_v[j_off+i]/(-dz[0]*dh)
		//				  -f_v[j_off+i];
		//	gg_v[0] = gg_v[0]/(tmp_a0-1.0f);
		//	
		//}else if (nbc == 2){
		//	//ee[0] = a[0]/(a[0]-1.0f);
		//	//gg[0] = dti2*(wfsurf[j_off+i]+rad[0]-rad[1])
		//	//	        /(dz[0]*dh) 
		//	//	    -f[j_off+i];
		//	//gg[0] = gg[0]/(a[0]-1.0f);

		//	//ee[0] = tmp_a0/(tmp_a0-1.0f);
		//	//gg[0] = dti2*(wfsurf[j_off+i]+rad[0]-rad[1])
		//	//	        /(dz[0]*dh) 
		//	//	    -f[j_off+i];
		//	//gg[0] = gg[0]/(tmp_a0-1.0f);

		//	ee_u[0] = tmp_a0/(tmp_a0-1.0f);
		//	gg_u[0] = dti2*(wfsurf_u[j_off+i]+rad[0]-rad[1])
		//		        /(dz[0]*dh) 
		//			  -f_u[j_off+i];
		//	gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		//	ee_v[0] = ee_u[0];
		//	gg_v[0] = dti2*(wfsurf_v[j_off+i]+rad[0]-rad[1])
		//		        /(dz[0]*dh) 
		//			  -f_v[j_off+i];
		//	gg_v[0] = gg_v[0]/(tmp_a0-1.0f);

		//}else if (nbc == 3 || nbc == 4){
		//	//ee[0] = 0;
		//	//gg[0] = fsurf[j_off+i];

		//	ee_u[0] = 0;
		//	gg_u[0] = fsurf_u[j_off+i];

		//	ee_v[0] = 0;
		//	gg_v[0] = fsurf_v[j_off+i];
		//}

		if (nbc_u == 1){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);

			gg_u[0] = -dti2*wfsurf_u[j_off+i]/(-dz[0]*dh)
						  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 2){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);
			gg_u[0] = dti2*(wfsurf_u[j_off+i]+rad_u[0]-rad_u[1])
				        /(dz[0]*dh) 
					  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 3 || nbc_u == 4){
			ee_u[0] = 0;
			gg_u[0] = fsurf_u[j_off+i];
		}
		
		if (nbc_v == 1){
			ee_v[0] = ee_u[0];

			gg_v[0] = -dti2*wfsurf_v[j_off+i]/(-dz[0]*dh)
						  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);
			
		}else if (nbc_v == 2){
			ee_v[0] = ee_u[0];
			gg_v[0] = dti2*(wfsurf_v[j_off+i]+rad_v[0]-rad_v[1])
				        /(dz[0]*dh) 
					  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);

		}else if (nbc_v == 3 || nbc_v == 4){
			ee_v[0] = 0;
			gg_v[0] = fsurf_v[j_off+i];
		}

		float tmp_ak, tmp_ck;

		for (k = 1; k < kbm2; k++){
			//gg[k] = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])-1.0f);
			//ee[k] = a[k]*gg[k];
			//gg[k] = (c[k]*gg[k-1]-f[k_off+j_off+i]
			//		  +dti2*(rad[k]-rad[k+1])/(dh*dz[k]))*gg[k];

			tmp_ak = -dti2*(kh[k_A1_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k]*dh*dh);

			tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k-1]*dh*dh);

			//gg[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee[k-1])-1.0f);
			//ee[k] = tmp_ak*gg[k];
			//gg[k] = (tmp_ck*gg[k-1]-f[k_off+j_off+i]
			//		  +dti2*(rad[k]-rad[k+1])/(dh*dz[k]))*gg[k];

			gg_u[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_u[k-1])-1.0f);
			ee_u[k] = tmp_ak*gg_u[k];
			gg_u[k] = (tmp_ck*gg_u[k-1]-f_u[k_off+j_off+i]
					  +dti2*(rad_u[k]-rad_u[k+1])/(dh*dz[k]))*gg_u[k];

			gg_v[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_v[k-1])-1.0f);
			ee_v[k] = tmp_ak*gg_v[k];
			gg_v[k] = (tmp_ck*gg_v[k-1]-f_v[k_off+j_off+i]
					  +dti2*(rad_v[k]-rad_v[k+1])/(dh*dz[k]))*gg_v[k];
		}

		tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
		  		/(dz[k]*dzz[k-1]*dh*dh);

		//f[kbm1_1_off+j_off+i] = (c[kbm1-1]*gg[kbm2-1]
		//						  -f[kbm1_1_off+j_off+i]
		//						  +dti2*(rad[kbm1-1]-rad[kb-1])
		//							   /(dh*dz[kbm1-1]))
		//					   /(c[kbm1-1]*(1.0f-ee[kbm2-1])-1.0f);

		//f[kbm1_1_off+j_off+i] = (tmp_ck*gg[kbm2-1]
		//						  -f[kbm1_1_off+j_off+i]
		//						  +dti2*(rad[kbm1-1]-rad[kb-1])
		//							   /(dh*dz[kbm1-1]))
		//					   /(tmp_ck*(1.0f-ee[kbm2-1])-1.0f);

		f_u[kbm1_1_off+j_off+i] = (tmp_ck*gg_u[kbm2-1]
								  -f_u[kbm1_1_off+j_off+i]
								  +dti2*(rad_u[kbm1-1]-rad_u[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_u[kbm2-1])-1.0f);

		f_v[kbm1_1_off+j_off+i] = (tmp_ck*gg_v[kbm2-1]
								  -f_v[kbm1_1_off+j_off+i]
								  +dti2*(rad_v[kbm1-1]-rad_v[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_v[kbm2-1])-1.0f);

		for (ki = kb-3; ki >= 0; ki--){
			//f[ki_off+j_off+i] = (ee[ki]*f[ki_A1_off+j_off+i]+gg[ki]);	
			f_u[ki_off+j_off+i] = (ee_u[ki]*f_u[ki_A1_off+j_off+i]
									+gg_u[ki]);	
			f_v[ki_off+j_off+i] = (ee_v[ki]*f_v[ki_A1_off+j_off+i]
									+gg_v[ki]);	
		}
	}
}

void proft_fusion_gpu(
				float *d_f_u, float *d_wfsurf_u,
				float *d_fsurf_u, int nbc_u,
				float *d_f_v, float *d_wfsurf_v,
				float *d_fsurf_v, int nbc_v){
	//modify:
	//		+ f

#ifndef TIME_DISABLE
	struct timeval start_fusion_proft,
				   end_fusion_proft;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_fusion_proft);
#endif


	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
					  (j_size+block_j_2D-1)/block_j_2D);

	proft_fusion_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_kh, d_etf, d_swrad, 
			d_wfsurf_u, d_f_u, d_fsurf_u,
			d_wfsurf_v, d_f_v, d_fsurf_v,
			d_h, d_dz, d_dzz, d_z, 
			dti2, umol, ntp, nbc_u, nbc_v,
			kb, jm, im);

#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_fusion_proft);
		proft_fusion_time += time_consumed(&start_fusion_proft, 
									&end_fusion_proft);
#endif

    return;
}


__global__ void
advuv_fusion_gpu_kernel_0(
				  float * __restrict__ uf, 
			      float * __restrict__ vf, 
				  const float * __restrict__ ub, 
				  const float * __restrict__ vb,
				  const float * __restrict__ u, 
				  const float * __restrict__ v, 
				  const float * __restrict__ w,
				  const float * __restrict__ advx, 
				  const float * __restrict__ advy,
				  const float * __restrict__ egf, 
				  const float * __restrict__ egb, 
				  const float * __restrict__ etf, 
				  const float * __restrict__ etb,
				  const float * __restrict__ dt, 
				  const float * __restrict__ e_atmos, 
				  const float * __restrict__ drhox,
				  const float * __restrict__ drhoy,
				  const float * __restrict__ h, 
				  const float * __restrict__ cor, 
				  const float * __restrict__ aru, 
				  const float * __restrict__ arv, 
				  const float * __restrict__ dx, 
				  const float * __restrict__ dy, 
				  const float * __restrict__ dz, 
				  float grav, float dti2, 
				  int kb, int jm, int im){

	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm2 = kb-2;
	int jmm1 = jm-1;
	int imm1 = im-1;

	if (j < jm && i < im){
		uf[j_off+i] = 0;	
		vf[j_off+i] = 0;	
		uf[kb_1_off+j_off+i] = 0;	
		vf[kb_1_off+j_off+i] = 0;	
	}

	if (j > 0 && j < jmm1 && i > 0 && i < imm1){

		for (k = 0; k < kbm2; k++){
			uf[k_A1_off+j_off+i] = 0.25f*(w[k_A1_off+j_off+i]
										+w[k_A1_off+j_off+(i-1)])
									 *(u[k_A1_off+j_off+i]
										+u[k_off+j_off+i]);	


			uf[k_off+j_off+i] = advx[k_off+j_off+i]
							   +(uf[k_off+j_off+i]
									-uf[k_A1_off+j_off+i])
								 *aru[j_off+i]/dz[k]
							   -aru[j_off+i]*0.25f
								 *(cor[j_off+i]*dt[j_off+i]
									*(v[k_off+j_A1_off+i]
										+v[k_off+j_off+i])
								  +cor[j_off+(i-1)]*dt[j_off+(i-1)]
									*(v[k_off+j_A1_off+(i-1)]
										+v[k_off+j_off+(i-1)]))
							   +grav*0.125f
								 *(dt[j_off+i]+dt[j_off+(i-1)])
								 *(egf[j_off+i]-egf[j_off+(i-1)]
									+egb[j_off+i]-egb[j_off+(i-1)]
									+(e_atmos[j_off+i]
										-e_atmos[j_off+(i-1)])*2.0f)
								 *(dy[j_off+i]+dy[j_off+(i-1)])
							   +drhox[k_off+j_off+i];

			uf[k_off+j_off+i] = ((h[j_off+i]
									+etb[j_off+i]
									+h[j_off+(i-1)]
									+etb[j_off+(i-1)])
								  *aru[j_off+i]*ub[k_off+j_off+i]
								 -2.0f*dti2*uf[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i]
									+h[j_off+(i-1)]+etf[j_off+(i-1)])
								  *aru[j_off+i]);

			/////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////

			vf[k_A1_off+j_off+i] = 0.25f*(w[k_A1_off+j_off+i]
										+w[k_A1_off+j_1_off+i])
									 *(v[k_A1_off+j_off+i]
										+v[k_off+j_off+i]);	

			vf[k_off+j_off+i] = advy[k_off+j_off+i]
							   +(vf[k_off+j_off+i]
									-vf[k_A1_off+j_off+i])
								  *arv[j_off+i]/dz[k]
							   +arv[j_off+i]*0.25f
								  *(cor[j_off+i]*dt[j_off+i]
									 *(u[k_off+j_off+(i+1)]
										 +u[k_off+j_off+i])
								   +cor[j_1_off+i]*dt[j_1_off+i]
									 *(u[k_off+j_1_off+(i+1)]
										 +u[k_off+j_1_off+i]))
							   +grav*0.125f*(dt[j_off+i]+dt[j_1_off+i])
								  *(egf[j_off+i]-egf[j_1_off+i]
									 +egb[j_off+i]-egb[j_1_off+i]
									 +(e_atmos[j_off+i]
										 -e_atmos[j_1_off+i])*2.0f)
								  *(dx[j_off+i]+dx[j_1_off+i])
							   +drhoy[k_off+j_off+i];
			
			vf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i]
										+h[j_1_off+i]+etb[j_1_off+i])
									*arv[j_off+i]*vb[k_off+j_off+i]
								  -2.0f*dti2*vf[k_off+j_off+i])
								/((h[j_off+i]+etf[j_off+i]
									+h[j_1_off+i]+etf[j_1_off+i])
								  *arv[j_off+i]);
	
		}

		uf[k_off+j_off+i] = advx[k_off+j_off+i]
						   +(uf[k_off+j_off+i]
								-uf[k_A1_off+j_off+i])
							 *aru[j_off+i]/dz[k]
						   -aru[j_off+i]*0.25f
							 *(cor[j_off+i]*dt[j_off+i]
								*(v[k_off+j_A1_off+i]
									+v[k_off+j_off+i])
							  +cor[j_off+(i-1)]*dt[j_off+(i-1)]
								*(v[k_off+j_A1_off+(i-1)]
									+v[k_off+j_off+(i-1)]))
						   +grav*0.125f
							 *(dt[j_off+i]+dt[j_off+(i-1)])
							 *(egf[j_off+i]-egf[j_off+(i-1)]
								+egb[j_off+i]-egb[j_off+(i-1)]
								+(e_atmos[j_off+i]
									-e_atmos[j_off+(i-1)])*2.0f)
							 *(dy[j_off+i]+dy[j_off+(i-1)])
						   +drhox[k_off+j_off+i];
		
	
		uf[k_off+j_off+i] = ((h[j_off+i]
								+etb[j_off+i]
								+h[j_off+(i-1)]
								+etb[j_off+(i-1)])
							  *aru[j_off+i]*ub[k_off+j_off+i]
							 -2.0f*dti2*uf[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i]
								+h[j_off+(i-1)]+etf[j_off+(i-1)])
							  *aru[j_off+i]);

		vf[k_off+j_off+i] = advy[k_off+j_off+i]
						   +(vf[k_off+j_off+i]
								-vf[k_A1_off+j_off+i])
							  *arv[j_off+i]/dz[k]
						   +arv[j_off+i]*0.25f
							  *(cor[j_off+i]*dt[j_off+i]
								 *(u[k_off+j_off+(i+1)]
									 +u[k_off+j_off+i])
							   +cor[j_1_off+i]*dt[j_1_off+i]
								 *(u[k_off+j_1_off+(i+1)]
									 +u[k_off+j_1_off+i]))
						   +grav*0.125f*(dt[j_off+i]+dt[j_1_off+i])
							  *(egf[j_off+i]-egf[j_1_off+i]
								 +egb[j_off+i]-egb[j_1_off+i]
								 +(e_atmos[j_off+i]
									 -e_atmos[j_1_off+i])*2.0f)
							  *(dx[j_off+i]+dx[j_1_off+i])
						   +drhoy[k_off+j_off+i];

		vf[k_off+j_off+i] = ((h[j_off+i]+etb[j_off+i]
									+h[j_1_off+i]+etb[j_1_off+i])
								*arv[j_off+i]*vb[k_off+j_off+i]
							  -2.0f*dti2*vf[k_off+j_off+i])
							/((h[j_off+i]+etf[j_off+i]
								+h[j_1_off+i]+etf[j_1_off+i])
							  *arv[j_off+i]);
	}

	return;

}

void advuv_fusion_gpu(){

	//modify: - uf
	//comments: in GPU version ,we ignore the value assigned for uf on j=0,
	//          I believe these values will be set later by MPI communication 
	//comments: above is wrong, after a test in 2013/07/14/, if we don't set 
	//			proper values on the boundary, the result will not concide with
	//			the C/Fortran version
	//comment: ?? I forgot about above 2013/07/25

#ifndef TIME_DISABLE
	struct timeval start_advuv_fusion,
				   end_advuv_fusion;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_advuv_fusion);
#endif

	dim3 threadPerBlock(block_i_2D, block_j_2D);
	dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
					  (j_size+block_j_2D-1)/block_j_2D);

	advuv_fusion_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_uf, d_vf, d_ub, d_vb, d_u, d_v, d_w, 
			d_advx, d_advy, d_egf, d_egb, d_etf, d_etb,
			d_dt, d_e_atmos, d_drhox, d_drhoy, 
			d_h, d_cor, d_aru, d_arv, d_dx, d_dy, d_dz,
			grav, dti2, kb, jm, im);

	
#ifndef TIME_DISABLE
		checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_advuv_fusion);
		advuv_fusion_time += time_consumed(&start_advuv_fusion, 
										   &end_advuv_fusion);
#endif
	
	return;
}

__global__ void
vertvl_overlap_bcond_gpu_kernel_0(float * __restrict__ xflux, 
					float * __restrict__ yflux, 
				    const float * __restrict__ u, 
					const float * __restrict__ v, 
					const float * __restrict__ dt,
					const float * __restrict__ dx, 
					const float * __restrict__ dy, 
				    int kb, int jm, int im){

	//only modified -w 
	int k;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	int kbm1 = kb-1;

	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				xflux[k][j][i] = 0.25f*(dy[j][i]+dy[j][i-1])
								  *(dt[j][i]+dt[j][i-1])*u[k][j][i];
			}
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			xflux[k_off+j_off+i] = 0.25f*(dy[j_off+i]+dy[j_off+(i-1)])
										*(dt[j_off+i]+dt[j_off+(i-1)])
										*u[k_off+j_off+i];
		}
	}

	/*
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				yflux[k][j][i] = 0.25f*(dx[j][i]+dx[j-1][i])
								  *(dt[j][i]+dt[j-1][i])*v[k][j][i];
			}
		}
	}
	*/

	if (j > 0 && j < jm && i > 0 && i < im){
		for (k = 0; k < kbm1; k++){
			yflux[k_off+j_off+i] = 0.25f*(dx[j_off+i]+dx[j_1_off+i])
									    *(dt[j_off+i]+dt[j_1_off+i])
										*v[k_off+j_off+i];
		}
	}

}
__global__ void
vertvl_overlap_bcond_inner_gpu_kernel_1(float * __restrict__ w, 
					const float * __restrict__ vfluxb, 
					const float * __restrict__ vfluxf,
		            const float * __restrict__ xflux, 
					const float * __restrict__ yflux, 
					const float * __restrict__ etf, 
					const float * __restrict__ etb, 
					const float * __restrict__ fsm, 
				    const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const float * __restrict__ dz,
				    float dti2, int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 33;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 33;

	if (j < jm-33 && i < im-33){
		w[j_off+i] = 0.5f*(vfluxb[j_off+i]+vfluxf[j_off+i]);

		for (k = 0; k < kb-1; k++){
			w[k_A1_off+j_off+i] = w[k_off+j_off+i]
								 +dz[k]*((xflux[k_off+j_off+(i+1)]
										   -xflux[k_off+j_off+i]
										   +yflux[k_off+j_A1_off+i]
										   -yflux[k_off+j_off+i]) 
										 /(dx[j_off+i]*dy[j_off+i]) 
										+(etf[j_off+i]-etb[j_off+i])/dti2);	

			w[k_off+j_off+i] *= fsm[j_off+i];
		}
		w[k_off+j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
vertvl_overlap_bcond_ew_gpu_kernel_1(float * __restrict__ w, 
					const float * __restrict__ vfluxb, 
					const float * __restrict__ vfluxf,
		            const float * __restrict__ xflux, 
					const float * __restrict__ yflux, 
					const float * __restrict__ etf, 
					const float * __restrict__ etb, 
				    const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const float * __restrict__ dz,
				    float dti2, int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	if (j < jm-1){
		w[j_off+i] = 0.5f*(vfluxb[j_off+i]+vfluxf[j_off+i]);

		for (k = 0; k < kb-1; k++){
			w[k_A1_off+j_off+i] = w[k_off+j_off+i]
								 +dz[k]*((xflux[k_off+j_off+(i+1)]
										   -xflux[k_off+j_off+i]
										   +yflux[k_off+j_A1_off+i]
										   -yflux[k_off+j_off+i]) 
										 /(dx[j_off+i]*dy[j_off+i]) 
										+(etf[j_off+i]-etb[j_off+i])/dti2);	
		}
	}
}

__global__ void
vertvl_overlap_bcond_sn_gpu_kernel_1(float * __restrict__ w, 
					const float * __restrict__ vfluxb, 
					const float * __restrict__ vfluxf,
		            const float * __restrict__ xflux, 
					const float * __restrict__ yflux, 
					const float * __restrict__ etf, 
					const float * __restrict__ etb, 
				    const float * __restrict__ dx, 
					const float * __restrict__ dy, 
					const float * __restrict__ dz,
				    float dti2, int kb, int jm, int im){

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){
		w[j_off+i] = 0.5f*(vfluxb[j_off+i]+vfluxf[j_off+i]);

		for (k = 0; k < kb-1; k++){
			w[k_A1_off+j_off+i] = w[k_off+j_off+i]
								 +dz[k]*((xflux[k_off+j_off+(i+1)]
										   -xflux[k_off+j_off+i]
										   +yflux[k_off+j_A1_off+i]
										   -yflux[k_off+j_off+i]) 
										 /(dx[j_off+i]*dy[j_off+i]) 
										+(etf[j_off+i]-etb[j_off+i])/dti2);	
		}
	}
}

void vertvl_overlap_bcond(cudaStream_t &stream_inner,
						cudaStream_t &stream_ew,
						cudaStream_t &stream_sn){

#ifndef TIME_DISABLE
	struct timeval start_vertvl,
				   end_vertvl;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_vertvl);
#endif

	//modified -w 
	//NO! the boundary is useful!, so we need copy-in w
	//int i,j,k;

	//dim3 threadPerBlock(block_i_2D, block_j_2D);
	//dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);

	//dim3 threadPerBlock_inner(block_i_2D, block_j_2D);
	//dim3 blockPerGrid_inner((i_size-2+block_i_2D-1)/block_i_2D, 
	//						(j_size-2+block_j_2D-1)/block_j_2D);

	//dim3 threadPerBlock_ew(32, 4);
	//dim3 blockPerGrid_ew(2, (j_size-2+3)/4);

	//dim3 threadPerBlock_sn(32, 4);
	//dim3 blockPerGrid_sn((i_size-2+31)/32, 16);

	/*
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	*/

    float *d_xflux = d_3d_tmp0;
	float *d_yflux = d_3d_tmp1;
	
	vertvl_overlap_bcond_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
			d_xflux, d_yflux, d_u, d_v, d_dt,
			d_dx, d_dy, kb, jm, im);

	vertvl_overlap_bcond_ew_gpu_kernel_1<<<blockPerGrid_ew_32, 
									 threadPerBlock_ew_32, 
									 0, stream_ew>>>(
			d_w, d_vfluxb, d_vfluxf,
		    d_xflux, d_yflux, d_etf, d_etb, 
			d_dx, d_dy, d_dz,
			dti2, kb, jm, im);

	vertvl_overlap_bcond_sn_gpu_kernel_1<<<blockPerGrid_sn_32, 
									 threadPerBlock_sn_32, 
									 0, stream_sn>>>(
			d_w, d_vfluxb, d_vfluxf,
		    d_xflux, d_yflux, d_etf, d_etb, 
			d_dx, d_dy, d_dz,
			dti2, kb, jm, im);

	vertvl_overlap_bcond_inner_gpu_kernel_1<<<blockPerGrid_inner, 
										threadPerBlock_inner, 
										0, stream_inner>>>(
			d_w, d_vfluxb, d_vfluxf,
		    d_xflux, d_yflux, d_etf, d_etb, 
			d_fsm, d_dx, d_dy, d_dz,
			dti2, kb, jm, im);

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_vertvl);
		vertvl_time += time_consumed(&start_vertvl, 
									 &end_vertvl);
#endif

    return;
}

__global__ void
proft_fusion_overlap_bcond_inner_gpu_kernel_0(
				   float * __restrict__ kh, 
				   float * __restrict__ etf, 
				   float * __restrict__ swrad,
				   float * __restrict__ wfsurf_u, 
				   float * __restrict__ f_u, 
				   const float * __restrict__ fsurf_u,
				   float * __restrict__ wfsurf_v, 
				   float * __restrict__ f_v, 
				   const float * __restrict__ fsurf_v,
				   const float * __restrict__ fsm, 
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz, 
				   const float * __restrict__ z,
				   float dti2, float umol, int ntp, 
				   int nbc_u, int nbc_v,
				   int kb, int jm, int im){

	//		+ f
	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+33;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+33;

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};

	float dh;
	//float a[k_size], c[k_size];
	//float ee[k_size], gg[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];
	float rad_u[k_size], rad_v[k_size];

	if (j < jm-33 && i < im-33){
		dh = h[j_off+i]+etf[j_off+i];	

		for (k = 0; k < kb; k++){
			rad_u[k] = 0;	
			rad_v[k] = 0;	
		}
	
		if (nbc_u == 2 || nbc_u == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_u[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		if (nbc_v == 2 || nbc_v == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_v[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		float tmp_a0 = -dti2*(kh[jm*im+j_off+i]+umol)
					        /(dz[0]*dzz[0]*dh*dh);


		if (nbc_u == 1){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);

			gg_u[0] = -dti2*wfsurf_u[j_off+i]/(-dz[0]*dh)
						  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 2){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);
			gg_u[0] = dti2*(wfsurf_u[j_off+i]+rad_u[0]-rad_u[1])
				        /(dz[0]*dh) 
					  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 3 || nbc_u == 4){
			ee_u[0] = 0;
			gg_u[0] = fsurf_u[j_off+i];
		}
		
		if (nbc_v == 1){
			ee_v[0] = ee_u[0];

			gg_v[0] = -dti2*wfsurf_v[j_off+i]/(-dz[0]*dh)
						  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);
			
		}else if (nbc_v == 2){
			ee_v[0] = ee_u[0];
			gg_v[0] = dti2*(wfsurf_v[j_off+i]+rad_v[0]-rad_v[1])
				        /(dz[0]*dh) 
					  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);

		}else if (nbc_v == 3 || nbc_v == 4){
			ee_v[0] = 0;
			gg_v[0] = fsurf_v[j_off+i];
		}

		float tmp_ak, tmp_ck;

		for (k = 1; k < kbm2; k++){
			tmp_ak = -dti2*(kh[k_A1_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k]*dh*dh);

			tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k-1]*dh*dh);

			gg_u[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_u[k-1])-1.0f);
			ee_u[k] = tmp_ak*gg_u[k];
			gg_u[k] = (tmp_ck*gg_u[k-1]-f_u[k_off+j_off+i]
					  +dti2*(rad_u[k]-rad_u[k+1])/(dh*dz[k]))*gg_u[k];

			gg_v[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_v[k-1])-1.0f);
			ee_v[k] = tmp_ak*gg_v[k];
			gg_v[k] = (tmp_ck*gg_v[k-1]-f_v[k_off+j_off+i]
					  +dti2*(rad_v[k]-rad_v[k+1])/(dh*dz[k]))*gg_v[k];
		}

		tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
		  		/(dz[k]*dzz[k-1]*dh*dh);

		f_u[kbm1_1_off+j_off+i] = (tmp_ck*gg_u[kbm2-1]
								  -f_u[kbm1_1_off+j_off+i]
								  +dti2*(rad_u[kbm1-1]-rad_u[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_u[kbm2-1])-1.0f);

		f_v[kbm1_1_off+j_off+i] = (tmp_ck*gg_v[kbm2-1]
								  -f_v[kbm1_1_off+j_off+i]
								  +dti2*(rad_v[kbm1-1]-rad_v[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_v[kbm2-1])-1.0f);

		for (ki = kb-3; ki >= 0; ki--){
			f_u[ki_off+j_off+i] = (ee_u[ki]*f_u[ki_A1_off+j_off+i]
									+gg_u[ki]);

			f_u[ki_A1_off+j_off+i] *= fsm[j_off+i];

			f_v[ki_off+j_off+i] = (ee_v[ki]*f_v[ki_A1_off+j_off+i]
									+gg_v[ki])*fsm[j_off+i];	

			f_v[ki_A1_off+j_off+i] *= fsm[j_off+i];
		}

		f_u[j_off+i] *= fsm[j_off+i];
		f_v[j_off+i] *= fsm[j_off+i];
	}
}

__global__ void
proft_fusion_overlap_bcond_ew_gpu_kernel_0(float * __restrict__ kh, 
				   float * __restrict__ etf, 
				   float * __restrict__ swrad,
				   float * __restrict__ wfsurf_u, 
				   float * __restrict__ f_u, 
				   const float * __restrict__ fsurf_u,
				   float * __restrict__ wfsurf_v, 
				   float * __restrict__ f_v, 
				   const float * __restrict__ fsurf_v,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz, 
				   const float * __restrict__ z,
				   float dti2, float umol, int ntp, 
				   int nbc_u, int nbc_v,
				   int kb, int jm, int im){

	//		+ f
	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};

	float dh;
	//float a[k_size], c[k_size];
	//float ee[k_size], gg[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];
	float rad_u[k_size], rad_v[k_size];

	if (j < jm-1){
		dh = h[j_off+i]+etf[j_off+i];	

		for (k = 0; k < kb; k++){
			rad_u[k] = 0;	
			rad_v[k] = 0;	
		}
	
		if (nbc_u == 2 || nbc_u == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_u[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		if (nbc_v == 2 || nbc_v == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_v[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		float tmp_a0 = -dti2*(kh[jm*im+j_off+i]+umol)
					        /(dz[0]*dzz[0]*dh*dh);


		if (nbc_u == 1){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);

			gg_u[0] = -dti2*wfsurf_u[j_off+i]/(-dz[0]*dh)
						  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 2){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);
			gg_u[0] = dti2*(wfsurf_u[j_off+i]+rad_u[0]-rad_u[1])
				        /(dz[0]*dh) 
					  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 3 || nbc_u == 4){
			ee_u[0] = 0;
			gg_u[0] = fsurf_u[j_off+i];
		}
		
		if (nbc_v == 1){
			ee_v[0] = ee_u[0];

			gg_v[0] = -dti2*wfsurf_v[j_off+i]/(-dz[0]*dh)
						  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);
			
		}else if (nbc_v == 2){
			ee_v[0] = ee_u[0];
			gg_v[0] = dti2*(wfsurf_v[j_off+i]+rad_v[0]-rad_v[1])
				        /(dz[0]*dh) 
					  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);

		}else if (nbc_v == 3 || nbc_v == 4){
			ee_v[0] = 0;
			gg_v[0] = fsurf_v[j_off+i];
		}

		float tmp_ak, tmp_ck;

		for (k = 1; k < kbm2; k++){
			tmp_ak = -dti2*(kh[k_A1_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k]*dh*dh);

			tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k-1]*dh*dh);

			gg_u[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_u[k-1])-1.0f);
			ee_u[k] = tmp_ak*gg_u[k];
			gg_u[k] = (tmp_ck*gg_u[k-1]-f_u[k_off+j_off+i]
					  +dti2*(rad_u[k]-rad_u[k+1])/(dh*dz[k]))*gg_u[k];

			gg_v[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_v[k-1])-1.0f);
			ee_v[k] = tmp_ak*gg_v[k];
			gg_v[k] = (tmp_ck*gg_v[k-1]-f_v[k_off+j_off+i]
					  +dti2*(rad_v[k]-rad_v[k+1])/(dh*dz[k]))*gg_v[k];
		}

		tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
		  		/(dz[k]*dzz[k-1]*dh*dh);

		f_u[kbm1_1_off+j_off+i] = (tmp_ck*gg_u[kbm2-1]
								  -f_u[kbm1_1_off+j_off+i]
								  +dti2*(rad_u[kbm1-1]-rad_u[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_u[kbm2-1])-1.0f);

		f_v[kbm1_1_off+j_off+i] = (tmp_ck*gg_v[kbm2-1]
								  -f_v[kbm1_1_off+j_off+i]
								  +dti2*(rad_v[kbm1-1]-rad_v[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_v[kbm2-1])-1.0f);

		for (ki = kb-3; ki >= 0; ki--){
			f_u[ki_off+j_off+i] = (ee_u[ki]*f_u[ki_A1_off+j_off+i]
									+gg_u[ki]);	
			f_v[ki_off+j_off+i] = (ee_v[ki]*f_v[ki_A1_off+j_off+i]
									+gg_v[ki]);	
		}
	}
}

__global__ void
proft_fusion_overlap_bcond_sn_gpu_kernel_0(float * __restrict__ kh, 
				   float * __restrict__ etf, 
				   float * __restrict__ swrad,
				   float * __restrict__ wfsurf_u, 
				   float * __restrict__ f_u, 
				   const float * __restrict__ fsurf_u,
				   float * __restrict__ wfsurf_v, 
				   float * __restrict__ f_v, 
				   const float * __restrict__ fsurf_v,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz, 
				   const float * __restrict__ z,
				   float dti2, float umol, int ntp, 
				   int nbc_u, int nbc_v,
				   int kb, int jm, int im){

	//		+ f
	int k, ki;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};

	float dh;
	//float a[k_size], c[k_size];
	//float ee[k_size], gg[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];
	float rad_u[k_size], rad_v[k_size];

	if (i > 32 && i < im-33){ 
		dh = h[j_off+i]+etf[j_off+i];	

		for (k = 0; k < kb; k++){
			rad_u[k] = 0;	
			rad_v[k] = 0;	
		}
	
		if (nbc_u == 2 || nbc_u == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_u[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		if (nbc_v == 2 || nbc_v == 4){
			for (k = 0; k < kbm1; k++){ 
				rad_v[k] = swrad[j_off+i]
					 		*(r[ntp-1]*expf(z[k]*dh/ad1[ntp-1])
					 	 +(1.0f-r[ntp-1])
					 		*expf(z[k]*dh/ad2[ntp-1]));
			}
		}

		float tmp_a0 = -dti2*(kh[jm*im+j_off+i]+umol)
					        /(dz[0]*dzz[0]*dh*dh);


		if (nbc_u == 1){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);

			gg_u[0] = -dti2*wfsurf_u[j_off+i]/(-dz[0]*dh)
						  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 2){
			ee_u[0] = tmp_a0/(tmp_a0-1.0f);
			gg_u[0] = dti2*(wfsurf_u[j_off+i]+rad_u[0]-rad_u[1])
				        /(dz[0]*dh) 
					  -f_u[j_off+i];
			gg_u[0] = gg_u[0]/(tmp_a0-1.0f);

		}else if (nbc_u == 3 || nbc_u == 4){
			ee_u[0] = 0;
			gg_u[0] = fsurf_u[j_off+i];
		}
		
		if (nbc_v == 1){
			ee_v[0] = ee_u[0];

			gg_v[0] = -dti2*wfsurf_v[j_off+i]/(-dz[0]*dh)
						  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);
			
		}else if (nbc_v == 2){
			ee_v[0] = ee_u[0];
			gg_v[0] = dti2*(wfsurf_v[j_off+i]+rad_v[0]-rad_v[1])
				        /(dz[0]*dh) 
					  -f_v[j_off+i];
			gg_v[0] = gg_v[0]/(tmp_a0-1.0f);

		}else if (nbc_v == 3 || nbc_v == 4){
			ee_v[0] = 0;
			gg_v[0] = fsurf_v[j_off+i];
		}

		float tmp_ak, tmp_ck;

		for (k = 1; k < kbm2; k++){
			tmp_ak = -dti2*(kh[k_A1_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k]*dh*dh);

			tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
			  		/(dz[k]*dzz[k-1]*dh*dh);

			gg_u[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_u[k-1])-1.0f);
			ee_u[k] = tmp_ak*gg_u[k];
			gg_u[k] = (tmp_ck*gg_u[k-1]-f_u[k_off+j_off+i]
					  +dti2*(rad_u[k]-rad_u[k+1])/(dh*dz[k]))*gg_u[k];

			gg_v[k] = 1.0f/(tmp_ak+tmp_ck*(1.0f-ee_v[k-1])-1.0f);
			ee_v[k] = tmp_ak*gg_v[k];
			gg_v[k] = (tmp_ck*gg_v[k-1]-f_v[k_off+j_off+i]
					  +dti2*(rad_v[k]-rad_v[k+1])/(dh*dz[k]))*gg_v[k];
		}

		tmp_ck = -dti2*(kh[k_off+j_off+i]+umol)
		  		/(dz[k]*dzz[k-1]*dh*dh);

		f_u[kbm1_1_off+j_off+i] = (tmp_ck*gg_u[kbm2-1]
								  -f_u[kbm1_1_off+j_off+i]
								  +dti2*(rad_u[kbm1-1]-rad_u[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_u[kbm2-1])-1.0f);

		f_v[kbm1_1_off+j_off+i] = (tmp_ck*gg_v[kbm2-1]
								  -f_v[kbm1_1_off+j_off+i]
								  +dti2*(rad_v[kbm1-1]-rad_v[kb-1])
									   /(dh*dz[kbm1-1]))
							   /(tmp_ck*(1.0f-ee_v[kbm2-1])-1.0f);

		for (ki = kb-3; ki >= 0; ki--){
			f_u[ki_off+j_off+i] = (ee_u[ki]*f_u[ki_A1_off+j_off+i]
									+gg_u[ki]);	
			f_v[ki_off+j_off+i] = (ee_v[ki]*f_v[ki_A1_off+j_off+i]
									+gg_v[ki]);	
		}
	}
}

void proft_fusion_overlap_bcond(
				float *d_f_u, float *d_wfsurf_u,
				float *d_fsurf_u, int nbc_u,
				float *d_f_v, float *d_wfsurf_v,
				float *d_fsurf_v, int nbc_v,
				cudaStream_t &stream_inner,
				cudaStream_t &stream_ew,
				cudaStream_t &stream_sn){

	//modify:
	//		+ f

#ifndef TIME_DISABLE
	struct timeval start_fusion_proft,
				   end_fusion_proft;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_fusion_proft);
#endif


	//dim3 threadPerBlock(block_i_2D, block_j_2D);
	//dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, 
	//				  (j_size+block_j_2D-1)/block_j_2D);

	//proft_fusion_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		d_kh, d_etf, d_swrad, 
	//		d_wfsurf_u, d_f_u, d_fsurf_u,
	//		d_wfsurf_v, d_f_v, d_fsurf_v,
	//		d_h, d_dz, d_dzz, d_z, 
	//		dti2, umol, ntp, nbc_u, nbc_v,
	//		kb, jm, im);


	proft_fusion_overlap_bcond_ew_gpu_kernel_0<<<blockPerGrid_ew_32, 
								   threadPerBlock_ew_32,
								   0, stream_ew>>>(
			d_kh, d_etf, d_swrad, 
			d_wfsurf_u, d_f_u, d_fsurf_u,
			d_wfsurf_v, d_f_v, d_fsurf_v,
			d_h, d_dz, d_dzz, d_z, 
			dti2, umol, ntp, nbc_u, nbc_v,
			kb, jm, im);

	proft_fusion_overlap_bcond_sn_gpu_kernel_0<<<blockPerGrid_sn_32, 
								   threadPerBlock_sn_32,
								   0, stream_sn>>>(
			d_kh, d_etf, d_swrad, 
			d_wfsurf_u, d_f_u, d_fsurf_u,
			d_wfsurf_v, d_f_v, d_fsurf_v,
			d_h, d_dz, d_dzz, d_z, 
			dti2, umol, ntp, nbc_u, nbc_v,
			kb, jm, im);

	proft_fusion_overlap_bcond_inner_gpu_kernel_0<<<
									  blockPerGrid_inner, 
									  threadPerBlock_inner,
									  0, stream_inner>>>(
			d_kh, d_etf, d_swrad, 
			d_wfsurf_u, d_f_u, d_fsurf_u,
			d_wfsurf_v, d_f_v, d_fsurf_v,
			d_fsm, d_h, d_dz, d_dzz, d_z, 
			dti2, umol, ntp, nbc_u, nbc_v,
			kb, jm, im);

	//checkCudaErrors(cudaStreamSynchronize(stream_ew));
	//checkCudaErrors(cudaStreamSynchronize(stream_sn));

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_fusion_proft);
		proft_fusion_time += time_consumed(&start_fusion_proft, 
									&end_fusion_proft);
#endif

    return;
}


__global__ void
profuv_fusion_overlap_bcond_inner_gpu_kernel_0(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb,
				   const float * __restrict__ km, 
				   const float * __restrict__ etf,
				   const float * __restrict__ wusurf, 
				   const float * __restrict__ wvsurf, 
				   float * __restrict__ wubot, 
				   float * __restrict__ wvbot, 
				   const float * __restrict__ cbc, 
				   const float * __restrict__ dum,
				   const float * __restrict__ dvm,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float dti2, float umol,
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y+threadIdx.y+33;
	const int i = blockDim.x*blockIdx.x+threadIdx.x+33;


	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	int jmm1 = jm-1;
	int imm1 = im-1;

	float dh_u, dh_v, tps_u, tps_v;
	//float a[k_size], c[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];

	if (j < jm-33 && i < im-33){

		dh_u = 0.5f*(h[j_off+i]+etf[j_off+i]
					+h[j_off+(i-1)]+etf[j_off+(i-1)]);

		dh_v = 0.5f*(h[j_off+i]+etf[j_off+i]
					+h[j_1_off+i]+etf[j_1_off+i]);

		//////////////////////////////////////////////
		float a_tmp_u, c_tmp_u, c_tmp_next_u;

		float a_tmp_v, c_tmp_v, c_tmp_next_v;

		c_tmp_u = (km[jm*im+j_off+i]
				  +km[jm*im+j_off+(i-1)])*0.5f;

		c_tmp_v = (km[jm*im+j_off+i]
				  +km[jm*im+j_1_off+i])*0.5f;

		a_tmp_u = -dti2*(c_tmp_u+umol)
					/(dz[0]*dzz[0]*dh_u*dh_u);

		a_tmp_v = -dti2*(c_tmp_v+umol)
					/(dz[0]*dzz[0]*dh_v*dh_v);

		//////////////////////////////////////////////

		ee_u[0] = a_tmp_u/(a_tmp_u-1.0f);
		gg_u[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh_u)
					-uf[j_off+i])
				 /(a_tmp_u-1.0f);

		//////////////////////////////////////////////

		ee_v[0] = a_tmp_v/(a_tmp_v-1.0f);
		gg_v[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh_v)
					-vf[j_off+i])
				 /(a_tmp_v-1.0f);

		//////////////////////////////////////////////

		c_tmp_u = -dti2*(c_tmp_u+umol)
					/(dz[1]*dzz[0]*dh_u*dh_u);

		c_tmp_v = -dti2*(c_tmp_v+umol)
					/(dz[1]*dzz[0]*dh_v*dh_v);

		//////////////////////////////////////////////

		for (k = 1; k < kbm2; k++){
			c_tmp_next_u = (km[k_A1_off+j_off+i]
						   +km[k_A1_off+j_off+(i-1)])*0.5f;

			c_tmp_next_v = (km[k_A1_off+j_off+i]
						   +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp_u = -dti2*(c_tmp_next_u+umol)
						 /(dz[k]*dzz[k]*dh_u*dh_u);

			a_tmp_v = -dti2*(c_tmp_next_v+umol)
						/(dz[k]*dzz[k]*dh_v*dh_v);

			//////////////////////////////////////////////

			gg_u[k] = 1.0f/(a_tmp_u+c_tmp_u*(1.0f-ee_u[k-1])-1.0f);

			ee_u[k] = a_tmp_u*gg_u[k];

			gg_u[k] = (c_tmp_u*gg_u[k-1]-uf[k_off+j_off+i])*gg_u[k];

			//////////////////////////////////////////////
			gg_v[k] = 1.0f/(a_tmp_v+c_tmp_v*(1.0f-ee_v[k-1])-1.0f);

			ee_v[k] = a_tmp_v*gg_v[k];

			gg_v[k] = (c_tmp_v*gg_v[k-1]-vf[k_off+j_off+i])*gg_v[k];

			//////////////////////////////////////////////
			c_tmp_u = -dti2*(c_tmp_next_u+umol)
						/(dz[k+1]*dzz[k]*dh_u*dh_u);

			c_tmp_v = -dti2*(c_tmp_next_v+umol)
						/(dz[k+1]*dzz[k]*dh_v*dh_v);
		}

		//////////////////////////////////////////////
		float tps_tmp_u = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		float tps_tmp_v = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		/////////////////////////////////////////////////

		tps_u = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp_u*tps_tmp_u));

		tps_v = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
				  *sqrtf((vb[kbm1_1_off+j_off+i]
						   *vb[kbm1_1_off+j_off+i])
						+(tps_tmp_v*tps_tmp_v));

		/////////////////////////////////////////////////

		uf[kbm1_1_off+j_off+i] = (c_tmp_u*gg_u[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps_u*dti2/(-dz[kbm1-1]*dh_u)
								  -1.0f
								  -(ee_u[kbm2-1]-1.0f)*c_tmp_u);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		/////////////////////////////////////////////////

		vf[kbm1_1_off+j_off+i] = (c_tmp_v*gg_v[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps_v*dti2/(-dz[kbm1-1]*dh_v)
								  -1.0f
								  -(ee_v[kbm2-1]-1.0f)*c_tmp_v);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		/////////////////////////////////////////////////
		//the boundary that *dum has been done here!

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee_u[ki]*uf[ki_A1_off+j_off+i]
								 +gg_u[ki])*dum[j_off+i];


			vf[ki_off+j_off+i] = (ee_v[ki]*vf[ki_A1_off+j_off+i]
								 +gg_v[ki])*dvm[j_off+i];

		}

		/////////////////////////////////////////////////

		wubot[j_off+i] = -tps_u*uf[kbm1_1_off+j_off+i];
		wvbot[j_off+i] = -tps_v*vf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profuv_fusion_overlap_bcond_ew_gpu_kernel_0(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb,
				   const float * __restrict__ km, 
				   const float * __restrict__ etf,
				   const float * __restrict__ wusurf, 
				   const float * __restrict__ wvsurf, 
				   float * __restrict__ wubot, 
				   float * __restrict__ wvbot, 
				   const float * __restrict__ cbc, 
				   const float * __restrict__ dum,
				   const float * __restrict__ dvm,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float dti2, float umol,
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;


	float dh_u, dh_v, tps_u, tps_v;
	//float a[k_size], c[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];


	if (j < jm-1){

		dh_u = 0.5f*(h[j_off+i]+etf[j_off+i]
					+h[j_off+(i-1)]+etf[j_off+(i-1)]);

		dh_v = 0.5f*(h[j_off+i]+etf[j_off+i]
					+h[j_1_off+i]+etf[j_1_off+i]);

		//////////////////////////////////////////////
		float a_tmp_u, c_tmp_u, c_tmp_next_u;

		float a_tmp_v, c_tmp_v, c_tmp_next_v;

		c_tmp_u = (km[jm*im+j_off+i]
				  +km[jm*im+j_off+(i-1)])*0.5f;

		c_tmp_v = (km[jm*im+j_off+i]
				  +km[jm*im+j_1_off+i])*0.5f;

		a_tmp_u = -dti2*(c_tmp_u+umol)
					/(dz[0]*dzz[0]*dh_u*dh_u);

		a_tmp_v = -dti2*(c_tmp_v+umol)
					/(dz[0]*dzz[0]*dh_v*dh_v);

		//////////////////////////////////////////////

		ee_u[0] = a_tmp_u/(a_tmp_u-1.0f);
		gg_u[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh_u)
					-uf[j_off+i])
				 /(a_tmp_u-1.0f);

		//////////////////////////////////////////////

		ee_v[0] = a_tmp_v/(a_tmp_v-1.0f);
		gg_v[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh_v)
					-vf[j_off+i])
				 /(a_tmp_v-1.0f);

		//////////////////////////////////////////////

		c_tmp_u = -dti2*(c_tmp_u+umol)
					/(dz[1]*dzz[0]*dh_u*dh_u);

		c_tmp_v = -dti2*(c_tmp_v+umol)
					/(dz[1]*dzz[0]*dh_v*dh_v);

		//////////////////////////////////////////////

		for (k = 1; k < kbm2; k++){
			c_tmp_next_u = (km[k_A1_off+j_off+i]
						   +km[k_A1_off+j_off+(i-1)])*0.5f;

			c_tmp_next_v = (km[k_A1_off+j_off+i]
						   +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp_u = -dti2*(c_tmp_next_u+umol)
						 /(dz[k]*dzz[k]*dh_u*dh_u);

			a_tmp_v = -dti2*(c_tmp_next_v+umol)
						/(dz[k]*dzz[k]*dh_v*dh_v);

			//////////////////////////////////////////////

			gg_u[k] = 1.0f/(a_tmp_u+c_tmp_u*(1.0f-ee_u[k-1])-1.0f);

			ee_u[k] = a_tmp_u*gg_u[k];

			gg_u[k] = (c_tmp_u*gg_u[k-1]-uf[k_off+j_off+i])*gg_u[k];

			//////////////////////////////////////////////
			gg_v[k] = 1.0f/(a_tmp_v+c_tmp_v*(1.0f-ee_v[k-1])-1.0f);

			ee_v[k] = a_tmp_v*gg_v[k];

			gg_v[k] = (c_tmp_v*gg_v[k-1]-vf[k_off+j_off+i])*gg_v[k];

			//////////////////////////////////////////////
			c_tmp_u = -dti2*(c_tmp_next_u+umol)
						/(dz[k+1]*dzz[k]*dh_u*dh_u);

			c_tmp_v = -dti2*(c_tmp_next_v+umol)
						/(dz[k+1]*dzz[k]*dh_v*dh_v);
		}

		//////////////////////////////////////////////
		float tps_tmp_u = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		float tps_tmp_v = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		/////////////////////////////////////////////////

		tps_u = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp_u*tps_tmp_u));

		tps_v = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
				  *sqrtf((vb[kbm1_1_off+j_off+i]
						   *vb[kbm1_1_off+j_off+i])
						+(tps_tmp_v*tps_tmp_v));

		/////////////////////////////////////////////////

		uf[kbm1_1_off+j_off+i] = (c_tmp_u*gg_u[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps_u*dti2/(-dz[kbm1-1]*dh_u)
								  -1.0f
								  -(ee_u[kbm2-1]-1.0f)*c_tmp_u);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		/////////////////////////////////////////////////

		vf[kbm1_1_off+j_off+i] = (c_tmp_v*gg_v[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps_v*dti2/(-dz[kbm1-1]*dh_v)
								  -1.0f
								  -(ee_v[kbm2-1]-1.0f)*c_tmp_v);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		/////////////////////////////////////////////////

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee_u[ki]*uf[ki_A1_off+j_off+i]
								 +gg_u[ki])*dum[j_off+i];

			vf[ki_off+j_off+i] = (ee_v[ki]*vf[ki_A1_off+j_off+i]
								 +gg_v[ki])*dvm[j_off+i];
		}

		/////////////////////////////////////////////////

		wubot[j_off+i] = -tps_u*uf[kbm1_1_off+j_off+i];
		wvbot[j_off+i] = -tps_v*vf[kbm1_1_off+j_off+i];
	}
}

__global__ void
profuv_fusion_overlap_bcond_sn_gpu_kernel_0(
				   float * __restrict__ uf, 
				   float * __restrict__ vf, 
				   const float * __restrict__ ub, 
				   const float * __restrict__ vb,
				   const float * __restrict__ km, 
				   const float * __restrict__ etf,
				   const float * __restrict__ wusurf, 
				   const float * __restrict__ wvsurf, 
				   float * __restrict__ wubot, 
				   float * __restrict__ wvbot, 
				   const float * __restrict__ cbc, 
				   const float * __restrict__ dum,
				   const float * __restrict__ dvm,
				   const float * __restrict__ h, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float dti2, float umol,
				   int kb, int jm, int im){

	int k, ki;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	int kbm1 = kb-1;
	int kbm2 = kbm1-1;

	float dh_u, dh_v, tps_u, tps_v;
	//float a[k_size], c[k_size];
	float ee_u[k_size], gg_u[k_size];
	float ee_v[k_size], gg_v[k_size];


	if (i > 32 && i < im-33){ 

		dh_u = 0.5f*(h[j_off+i]+etf[j_off+i]
					+h[j_off+(i-1)]+etf[j_off+(i-1)]);

		dh_v = 0.5f*(h[j_off+i]+etf[j_off+i]
					+h[j_1_off+i]+etf[j_1_off+i]);

		//////////////////////////////////////////////
		float a_tmp_u, c_tmp_u, c_tmp_next_u;

		float a_tmp_v, c_tmp_v, c_tmp_next_v;

		c_tmp_u = (km[jm*im+j_off+i]
				  +km[jm*im+j_off+(i-1)])*0.5f;

		c_tmp_v = (km[jm*im+j_off+i]
				  +km[jm*im+j_1_off+i])*0.5f;

		a_tmp_u = -dti2*(c_tmp_u+umol)
					/(dz[0]*dzz[0]*dh_u*dh_u);

		a_tmp_v = -dti2*(c_tmp_v+umol)
					/(dz[0]*dzz[0]*dh_v*dh_v);

		//////////////////////////////////////////////

		ee_u[0] = a_tmp_u/(a_tmp_u-1.0f);
		gg_u[0] = (-dti2*wusurf[j_off+i]/(-dz[0]*dh_u)
					-uf[j_off+i])
				 /(a_tmp_u-1.0f);

		//////////////////////////////////////////////

		ee_v[0] = a_tmp_v/(a_tmp_v-1.0f);
		gg_v[0] = (-dti2*wvsurf[j_off+i]/(-dz[0]*dh_v)
					-vf[j_off+i])
				 /(a_tmp_v-1.0f);

		//////////////////////////////////////////////

		c_tmp_u = -dti2*(c_tmp_u+umol)
					/(dz[1]*dzz[0]*dh_u*dh_u);

		c_tmp_v = -dti2*(c_tmp_v+umol)
					/(dz[1]*dzz[0]*dh_v*dh_v);

		//////////////////////////////////////////////

		for (k = 1; k < kbm2; k++){
			c_tmp_next_u = (km[k_A1_off+j_off+i]
						   +km[k_A1_off+j_off+(i-1)])*0.5f;

			c_tmp_next_v = (km[k_A1_off+j_off+i]
						   +km[k_A1_off+j_1_off+i])*0.5f;

			a_tmp_u = -dti2*(c_tmp_next_u+umol)
						 /(dz[k]*dzz[k]*dh_u*dh_u);

			a_tmp_v = -dti2*(c_tmp_next_v+umol)
						/(dz[k]*dzz[k]*dh_v*dh_v);

			//////////////////////////////////////////////

			gg_u[k] = 1.0f/(a_tmp_u+c_tmp_u*(1.0f-ee_u[k-1])-1.0f);

			ee_u[k] = a_tmp_u*gg_u[k];

			gg_u[k] = (c_tmp_u*gg_u[k-1]-uf[k_off+j_off+i])*gg_u[k];

			//////////////////////////////////////////////
			gg_v[k] = 1.0f/(a_tmp_v+c_tmp_v*(1.0f-ee_v[k-1])-1.0f);

			ee_v[k] = a_tmp_v*gg_v[k];

			gg_v[k] = (c_tmp_v*gg_v[k-1]-vf[k_off+j_off+i])*gg_v[k];

			//////////////////////////////////////////////
			c_tmp_u = -dti2*(c_tmp_next_u+umol)
						/(dz[k+1]*dzz[k]*dh_u*dh_u);

			c_tmp_v = -dti2*(c_tmp_next_v+umol)
						/(dz[k+1]*dzz[k]*dh_v*dh_v);
		}

		//////////////////////////////////////////////
		float tps_tmp_u = 0.25f*(vb[kbm1_1_off+j_off+i]
						  +vb[kbm1_1_off+j_A1_off+i]
						  +vb[kbm1_1_off+j_off+(i-1)]
						  +vb[kbm1_1_off+j_A1_off+(i-1)]);

		float tps_tmp_v = 0.25f*(ub[kbm1_1_off+j_off+i]
						  +ub[kbm1_1_off+j_off+i+1]
						  +ub[kbm1_1_off+j_1_off+i]
						  +ub[kbm1_1_off+j_1_off+(i+1)]);

		/////////////////////////////////////////////////

		tps_u = 0.5f*(cbc[j_off+i]+cbc[j_off+(i-1)])
				  *sqrtf((ub[kbm1_1_off+j_off+i]
						   *ub[kbm1_1_off+j_off+i])
						+(tps_tmp_u*tps_tmp_u));

		tps_v = 0.5f*(cbc[j_off+i]+cbc[j_1_off+i])
				  *sqrtf((vb[kbm1_1_off+j_off+i]
						   *vb[kbm1_1_off+j_off+i])
						+(tps_tmp_v*tps_tmp_v));

		/////////////////////////////////////////////////

		uf[kbm1_1_off+j_off+i] = (c_tmp_u*gg_u[kbm2-1]
								  -uf[kbm1_1_off+j_off+i])
								/(tps_u*dti2/(-dz[kbm1-1]*dh_u)
								  -1.0f
								  -(ee_u[kbm2-1]-1.0f)*c_tmp_u);

		uf[kbm1_1_off+j_off+i] = uf[kbm1_1_off+j_off+i]*dum[j_off+i];

		/////////////////////////////////////////////////

		vf[kbm1_1_off+j_off+i] = (c_tmp_v*gg_v[kbm2-1]
								  -vf[kbm1_1_off+j_off+i])
								/(tps_v*dti2/(-dz[kbm1-1]*dh_v)
								  -1.0f
								  -(ee_v[kbm2-1]-1.0f)*c_tmp_v);

		vf[kbm1_1_off+j_off+i] = vf[kbm1_1_off+j_off+i]*dvm[j_off+i];

		/////////////////////////////////////////////////

		for (ki = kb-3; ki >= 0; ki--){
			uf[ki_off+j_off+i] = (ee_u[ki]*uf[ki_A1_off+j_off+i]
								 +gg_u[ki])*dum[j_off+i];

			vf[ki_off+j_off+i] = (ee_v[ki]*vf[ki_A1_off+j_off+i]
								 +gg_v[ki])*dvm[j_off+i];
		}

		/////////////////////////////////////////////////

		wubot[j_off+i] = -tps_u*uf[kbm1_1_off+j_off+i];
		wvbot[j_off+i] = -tps_v*vf[kbm1_1_off+j_off+i];
	}
}

void profuv_fusion_overlap_bcond(){

//modify +uf -wubot

#ifndef TIME_DISABLE
	struct timeval start_profuv_fusion,
				   end_profuv_fusion;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_profuv_fusion);
#endif


	//dim3 threadPerBlock(block_i_2D, block_j_2D);
	//dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D, (j_size+block_j_2D-1)/block_j_2D);


	//profuv_fusion_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//	    d_uf, d_vf, d_ub, d_vb,  d_km, d_etf, 
	//		d_wusurf, d_wvsurf, d_wubot, d_wvbot, 
	//		d_cbc, d_dum, d_dvm, d_h, d_dz, d_dzz, 
	//		dti2, umol, kb, jm, im);


	profuv_fusion_overlap_bcond_ew_gpu_kernel_0<<<
									blockPerGrid_ew_32, 
									threadPerBlock_ew_32,
									0, stream[1]>>>(
		    d_uf, d_vf, d_ub, d_vb,  d_km, d_etf, 
			d_wusurf, d_wvsurf, d_wubot, d_wvbot, 
			d_cbc, d_dum, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);
	profuv_fusion_overlap_bcond_sn_gpu_kernel_0<<<
									blockPerGrid_sn_32, 
									threadPerBlock_sn_32,
									0, stream[2]>>>(
		    d_uf, d_vf, d_ub, d_vb,  d_km, d_etf, 
			d_wusurf, d_wvsurf, d_wubot, d_wvbot, 
			d_cbc, d_dum, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);


	profuv_fusion_overlap_bcond_inner_gpu_kernel_0<<<
									   blockPerGrid_inner, 
									   threadPerBlock_inner,
									   0, stream[0]>>>(
		    d_uf, d_vf, d_ub, d_vb,  d_km, d_etf, 
			d_wusurf, d_wvsurf, d_wubot, d_wvbot, 
			d_cbc, d_dum, d_dvm, d_h, d_dz, d_dzz, 
			dti2, umol, kb, jm, im);


	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));
	//checkCudaErrors(cudaStreamSynchronize(stream[0]));

    //exchange2d_mpi_gpu(d_wubot,im,jm);
    //exchange2d_mpi_gpu(d_wvbot,im,jm);


    exchange2d_cudaUVA(d_wubot, d_wubot_east, d_wubot_west, 
					   d_wubot_south, d_wubot_north,
					   stream[1], im, jm);

    exchange2d_cudaUVA(d_wvbot, d_wvbot_east, d_wvbot_west, 
					   d_wvbot_south, d_wvbot_north,
					   stream[1], im, jm);

	//MPI_Barrier(pom_comm);
    //exchange2d_cuda_ipc(d_wubot, d_wubot_east, d_wubot_west, 
	//					stream[1], im, jm);
    //exchange2d_cuda_ipc(d_wvbot, d_wvbot_east, d_wvbot_west, 
	//					stream[1], im, jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//MPI_Barrier(pom_comm);
	
#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_profuv_fusion);
		profuv_fusion_time += time_consumed(&start_profuv_fusion, 
									&end_profuv_fusion);
#endif

    return;
}


//__global__ void
//profq_gpu_kernel_0(float * __restrict__ utau2, 
//				   float * __restrict__ uf, 
//				   const float * __restrict__ wusurf, 
//				   const float * __restrict__ wvsurf, 
//				   const float * __restrict__ wubot, 
//				   const float * __restrict__ wvbot,
//				   int kb, int jm, int im){
//
//	//int k;
//	int j = blockDim.y*blockIdx.y + threadIdx.y;
//	int i = blockDim.x*blockIdx.x + threadIdx.x;
//
//	//int kbm1 = kb-1;
//	int jmm1 = jm-1;
//	int imm1 = im-1;
//
//	float sef = 1.f;
//	float const1;
//
//
///*	
//! the following section solves the equation:
//!     dti2*(kq*q2')' - q2*(2.*dti2*dtef+1.) = -q2b
//*/
////! surface and bottom boundary conditions
//      //const1=(16.6e0**(2.e0/3.e0))*sef;
//    const1=powf(16.6e0f,(2.e0f/3.e0f))*sef;
//	  
//
//	/*
//	for(j = 0; j < jmm1; j++){
//		for(i = 0; i < imm1; i++){
//			float tmpu_surf = 0.5f*(wusurf[j][i]+wusurf[j][i+1]);
//			float tmpv_surf = 0.5f*(wvsurf[j][i]+wvsurf[j+1][i]);
//			utau2[j][i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
//
//			ee[0][j][i] = 0.0;
//			gg[0][j][i] = powf((15.8f*cbcnst),(2.0f/3.0f))*utau2[j][i];
//			//comment by xsz: powf (float)powf
//			l0[j][i] = surfl*utau2[j][i]/grav;
//
//			float tmpu_bot = 0.5f*(wubot[j][i]+wubot[j][i+1]);
//			float tmpv_bot = 0.5f*(wvbot[j][i]+wvbot[j+1][i]);
//
//			uf[kb-1][j][i] = sqrtf((tmpu_bot*tmpu_bot)
//								   +(tmpv_bot*tmpv_bot))
//							 *const1;
//		}
//	}
//	*/
//	
//	if (j < jmm1 && i < imm1){
//		float tmpu_surf = 0.5f*(wusurf[j_off+i]+wusurf[j_off+(i+1)]);
//		float tmpv_surf = 0.5f*(wvsurf[j_off+i]+wvsurf[j_A1_off+i]);
//		utau2[j_off+i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
//
//		//ee[j_off+i] = 0;
//		//gg[j_off+i] = powf((15.8f*cbcnst), (2.f/3.f))*utau2[j_off+i];
//		//l0[j_off+i] = surfl*utau2[j_off+i]/grav;
//
//		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
//		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
//
//		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
//									+(tmpv_bot*tmpv_bot))
//							  *const1;
//	}
//	//xsz_uf
//}

__global__ void
profq_overlap_bcond_inner_gpu_kernel_0(float * __restrict__ utau2, 
				   //float * __restrict__ uf, 
				   const float * __restrict__ wusurf, 
				   const float * __restrict__ wvsurf, 
				   //const float * __restrict__ wubot, 
				   //const float * __restrict__ wvbot,
				   int kb, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y+33;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+33;

	const int jmm1 = jm-1;
	const int imm1 = im-1;

	//const float sef = 1.f;
	//float const1;

    //const1=powf(16.6e0f,(2.e0f/3.e0f))*sef;

	if (j < jm-33 && i < im-33){
		float tmpu_surf = 0.5f*(wusurf[j_off+i]+wusurf[j_off+(i+1)]);
		float tmpv_surf = 0.5f*(wvsurf[j_off+i]+wvsurf[j_A1_off+i]);
		utau2[j_off+i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
	}

	//if (j < jmm1 && i < imm1){
	//	float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
	//	float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
	//	uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
	//								+(tmpv_bot*tmpv_bot))
	//						  *const1;
	//}
}

__global__ void
profq_overlap_bcond_ew_gpu_kernel_0(float * __restrict__ utau2, 
					  const float * __restrict__ wusurf, 
				      const float * __restrict__ wvsurf, 
				      int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	int i;
	
	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;
	}

	if (j < jm-1){
		float tmpu_surf = 0.5f*(wusurf[j_off+i]+wusurf[j_off+(i+1)]);
		float tmpv_surf = 0.5f*(wvsurf[j_off+i]+wvsurf[j_A1_off+i]);
		utau2[j_off+i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
	}
}

__global__ void
profq_overlap_bcond_sn_gpu_kernel_0(float * __restrict__ utau2, 
					  const float * __restrict__ wusurf, 
				      const float * __restrict__ wvsurf, 
					  int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	int j;
	
	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	if (i > 32 && i < im-33){
		float tmpu_surf = 0.5f*(wusurf[j_off+i]+wusurf[j_off+(i+1)]);
		float tmpv_surf = 0.5f*(wvsurf[j_off+i]+wvsurf[j_A1_off+i]);
		utau2[j_off+i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
	}
}

__global__ void
profq_overlap_bcond_ew_bcond_gpu_kernel_0(float * __restrict__ utau2, 
					  const float * __restrict__ wusurf, 
				      const float * __restrict__ wvsurf, 
				      int n_west, int jm, int im){

	const int j = blockDim.y*blockIdx.y + threadIdx.y;
	
	if (n_west == -1){
		if (j < jm-1){
			float tmpu_surf = 0.5f*(wusurf[j_off]+wusurf[j_off+1]);
			float tmpv_surf = 0.5f*(wvsurf[j_off]+wvsurf[j_A1_off]);
			utau2[j_off] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
		}
	}
}

__global__ void
profq_overlap_bcond_sn_bcond_gpu_kernel_0(float * __restrict__ utau2, 
					  const float * __restrict__ wusurf, 
				      const float * __restrict__ wvsurf, 
					  int n_south, int jm, int im){

	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j;

	if (n_south == -1){
		if (i > 0 && i < im-1){
			float tmpu_surf = 0.5f*(wusurf[i]+wusurf[(i+1)]);
			float tmpv_surf = 0.5f*(wvsurf[i]+wvsurf[im+i]);
			utau2[i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);
		}
	}
}


__global__ void
profq_overlap_bcond_inner_gpu_kernel_1(
				   const float * __restrict__ t, 
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
				   float * __restrict__ kq, 
				   float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ wubot, 
				   const float * __restrict__ wvbot,
				   const float * __restrict__ fsm, 
				   const float * __restrict__ h, 
				   const float * __restrict__ z, 
				   const float * __restrict__ zz, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   float umol, 
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 33;
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 33;

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

    float a1,a2,b1,b2,c1;
    float coef1,coef2,coef3,coef4,coef5;
    float e1,e2;
    float cbcnst, surfl, shiw;
    float p,sef,sp,tp;

    a1=0.92f;b1=16.6f;a2=0.74f;b2=10.1f;c1=0.08f;
    e1=1.8e0f;e2=1.33e0f;
    cbcnst=100.0f;surfl=2.0e5f;shiw=0.0f;
    sef=1.e0f;

	float const1;
    const1=powf(16.6e0f,(2.e0f/3.e0f))*sef;
    
	float ee_uf[k_size], ee_vf[k_size];
    float gg_uf[k_size], gg_vf[k_size];
    float stf;	

//	float a[k_size];
//	float c[k_size];
//	float ee[k_size];
//	float gg[k_size];
//	float sm[k_size];
//	float sh[k_size];
//	float cc[k_size];
//	float gh[k_size];
//	float boygr[k_size];
//	float stf;
//	float prod_0[k_size];
//	float prod[k_size];
//	float dtef[k_size];
//	float l[k_size];

	float dh, l0, utau2;
/*
	if (j < jm-33 && i < im-33){
		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];

		ee[0] = 0;
		gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		for (k = 0; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc[k] = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));
		}


		l[0] = kappa*l0;	
		l[kb-1] = 0;
		gh[0] = 0;
		gh[kb-1] = 0;


		stf = 1.0f;
		for (k = 1; k < kbm1; k++){
			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr[k] = grav*(rho[k_1_off+j_off+i]
							-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc[k-1]*cc[k-1])
						 +(cc[k]*cc[k]));

			l[k] = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l[k] = MAX(l[k], (kappa*l0));

			gh[k] = (l[k]*l[k])*boygr[k]
				   /(q2b[k_off+j_off+i]+small);

			gh[k] = MIN(gh[k], 0.028f);

			dtef[k] = sqrtf(ABS(q2b[k_off+j_off+i]))
						   *stf/(b1*l[k]+small);
		}

		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		for (k = 0; k < kb; k++){

			sh[k] = coef1/(1.0f-coef2*gh[k]);
			sm[k] = coef3+sh[k]*coef4*gh[k];
			sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

			prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
		}

		for (k = 1; k < kbm1; k++){
			a[k] = -dti2*(kq[k_A1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k]*dh*dh);

			c[k] = -dti2*(kq[k_1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k-1]*dh*dh);
			prod[k] = 0;

			kq[k_1_off+j_off+i] = (prod_0[k-1]*0.41f*sh[k-1]
								+kq[k_1_off+j_off+i])*0.5f;
		}

		kq[kb_2_off+j_off+i] = (prod_0[kb-2]*0.41f*sh[kb-2]
							   +kq[kb_2_off+j_off+i])*0.5f;
		kq[kb_1_off+j_off+i] = (prod_0[kb-1]*0.41f*sh[kb-1]
							   +kq[kb_1_off+j_off+i])*0.5f;


		for (k = 1; k < kbm1; k++){
			float tmpu = u[k_off+j_off+i]
						-u[k_1_off+j_off+i]
						+u[k_off+j_off+(i+1)]
						-u[k_1_off+j_off+(i+1)];

			float tmpv = v[k_off+j_off+i]
						-v[k_1_off+j_off+i]
						+v[k_off+j_A1_off+i]
						-v[k_1_off+j_A1_off+i];
			//////////////////////////////////
			//there is a little bug here,
			//but for 1 < k < kbm1, u and v will not cause 
			//segmentation fault~~
			//////////////////////////////////

			prod[k] = km[k_off+j_off+i]*0.25f*sef
					    *((tmpu*tmpu)+(tmpv*tmpv))
				        /((dzz[k-1]*dh)
					      *(dzz[k-1]*dh))
					 -shiw*km[k_off+j_off+i]
					    *boygr[k];

			prod[k] = prod[k]+kh[k_off+j_off+i]*boygr[k];

			km[k_off+j_off+i] = (prod_0[k]*sm[k]
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prod_0[k]*sh[k]
								+kh[k_off+j_off+i])*0.5f;
		}

		km[j_off+i] = (prod_0[0]*sm[0]
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prod_0[kb-1]*sm[kb-1]
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prod_0[0]*sh[0]
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prod_0[kb-1]*sh[kb-1]
							   +kh[kb_1_off+j_off+i])*0.5f;


		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
									+(tmpv_bot*tmpv_bot))
							  *const1;

		for (k = 1; k < kbm1; k++){
			float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
	   						 -(2.0f*dti2*dtef[k]+1.0f));

	   		ee[k] = a[k]*tmp;

	   		gg[k] = (-2.0f*dti2*prod[k]
	   				 +c[k]*gg[k-1]
	   				 -uf[k_off+j_off+i])*tmp;
		}

		///////////////////////////////////////////////////
		// in original code ABS(uf) is for k = 1 to k=kb-2
		// but uf[kb-1] is surely a positive number
		for (ki = kb-2; ki >= 0; ki--){
			uf[ki_off+j_off+i] = ee[ki]*uf[ki_A1_off+j_off+i]
								+gg[ki];	
			uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i])
								   *fsm[j_off+i];	
		}
		uf[j_off+i] *= fsm[j_off+i];


		vf[j_off+i] = 0;
		vf[kb_1_off+j_off+i] = 0;
		ee[1] = 0;
		gg[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
									*dh*q2[kbm1_1_off+j_off+i];

		for (k = 1; k < kbm1; k++){
			float tmp = (1.0f/ABS(z[k]-z[0])
							+1.0f/ABS(z[k]-z[kb-1]))
						*l[k]/(dh*kappa);

			tmp = 1.0f+e2*(tmp*tmp);
			dtef[k] *= tmp;
		}

		for (k = 2; k < kbm1; k++){
			float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
							 -(dti2*dtef[k]+1.0f));

			ee[k] = a[k]*tmp;

			gg[k] = (dti2*(-prod[k]*l[k]*e1)
					+c[k]*gg[k-1]
				    -vf[k_off+j_off+i])*tmp;
		}

		for (ki = kb-2; ki > 0; ki--){
			vf[ki_off+j_off+i] = ee[ki]*vf[ki_A1_off+j_off+i]
								+gg[ki];	

			vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i])*fsm[j_off+i];
		}
		vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i])*fsm[j_off+i];
		///////////////////////////////////////////
		//vf[0][j][i] = 0 and need not multiply fsm
	}
*/
	


















				   
	if (j < jm-33 && i < im-33){

		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];


		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		//l[0] = kappa*l0;	
		//l[kb-1] = 0;
		//gh[0] = 0;
		//gh[kb-1] = 0;
		stf = 1.0f;

		float cc_b, cc_n;
		float cc_tmp;
		tp = t[j_off+i] + tbias;
		sp = s[j_off+i] + sbias;
		p = grav*rhoref*(-zz[0]*h[j_off+i])*1.0e-4f;
		cc_tmp = 1449.1f+0.00821f*p
			   +4.55f*tp-0.045f*(tp*tp)
			   +1.34f*(sp-35.0f);

		cc_b = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
						   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

		float gh_surf = 0.0f, gh_bot = 0; 
		float gh_n;
		float l_surf = kappa*l0, l_bot = 0, l_n;
		float sh_surf = 0, sh_bot = 0; 
		float sh_b, sh_n;
		float sm_surf = 0, sm_bot = 0, sm_n;
		float prodk_surf = 0, prodk_bot = 0, prodk_b, prodk_n;
		float boygr_n, a_n, c_n, dtef_n, prod_n;

		ee_uf[0] = 0;
		gg_uf[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;

		vf[j_off+i] = 0;
		vf[kb_1_off+j_off+i] = 0;
		ee_vf[1] = 0;
		gg_vf[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
									*dh*q2[kbm1_1_off+j_off+i];


		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		sh_surf = coef1/(1.0f-coef2*gh_surf);
		sh_bot = coef1/(1.0f-coef2*gh_bot);
		sh_b = sh_surf;

		sm_surf = coef3+sh_surf*coef4*gh_surf;
		sm_surf = sm_surf/(1.0f-coef5*gh_surf); 
		sm_bot = coef3+sh_bot*coef4*gh_bot;
		sm_bot = sm_bot/(1.0f-coef5*gh_bot); 

		prodk_surf = l_surf*sqrtf(ABS(q2[j_off+i]));	
		prodk_bot = l_bot*sqrtf(ABS(q2[kb_1_off+j_off+i]));	
		prodk_b = prodk_surf;
		//prodk_b = prod_0[0];

		for (k = 1; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc_n = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr_n = grav*(rho[k_1_off+j_off+i]
						-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc_b*cc_b)
						 +(cc_n*cc_n));

			cc_b = cc_n;

			l_n = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l_n = MAX(l_n, (kappa*l0));

			gh_n = (l_n*l_n)*boygr_n
				   /(q2b[k_off+j_off+i]+small);

			gh_n = MIN(gh_n, 0.028f);

			dtef_n = sqrtf(ABS(q2b[k_off+j_off+i]))
						   *stf/(b1*l_n+small);

			sh_n = coef1/(1.0f-coef2*gh_n);
			sm_n = coef3+sh_n*coef4*gh_n;
			sm_n = sm_n/(1.0f-coef5*gh_n); 

			prodk_n= l_n*sqrtf(ABS(q2[k_off+j_off+i]));	

			a_n = -dti2*(kq[k_A1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k]*dh*dh);

			c_n = -dti2*(kq[k_1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k-1]*dh*dh);
			//prod[k] = 0;

			kq[k_1_off+j_off+i] = (prodk_b*0.41f*sh_b
								  +kq[k_1_off+j_off+i])*0.5f;

			float tmpu = u[k_off+j_off+i]
						-u[k_1_off+j_off+i]
						+u[k_off+j_off+(i+1)]
						-u[k_1_off+j_off+(i+1)];

			float tmpv = v[k_off+j_off+i]
						-v[k_1_off+j_off+i]
						+v[k_off+j_A1_off+i]
						-v[k_1_off+j_A1_off+i];

			prod_n = km[k_off+j_off+i]*0.25f*sef
					    *((tmpu*tmpu)+(tmpv*tmpv))
				        /((dzz[k-1]*dh)
					      *(dzz[k-1]*dh))
					 -shiw*km[k_off+j_off+i]
					    *boygr_n;

			prod_n = prod_n+kh[k_off+j_off+i]*boygr_n;

			km[k_off+j_off+i] = (prodk_n*sm_n
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prodk_n*sh_n
								+kh[k_off+j_off+i])*0.5f;

			sh_b = sh_n;
			prodk_b = prodk_n;



		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
									+(tmpv_bot*tmpv_bot))
							  *const1;


			/////////////////////////////
			/////////////////////////////

			float uf_tmp = 1.0f/(a_n+c_n*(1.0f-ee_uf[k-1])
	   						 -(2.0f*dti2*dtef_n+1.0f));

	   		ee_uf[k] = a_n*uf_tmp;

	   		gg_uf[k] = (-2.0f*dti2*prod_n
	   				 +c_n*gg_uf[k-1]
	   				 -uf[k_off+j_off+i])*uf_tmp;

			/////////////////////////////
			/////////////////////////////

			if (k > 1){
				float vf_tmp = (1.0f/ABS(z[k]-z[0])
								+1.0f/ABS(z[k]-z[kb-1]))
							*l_n/(dh*kappa);

				vf_tmp = 1.0f+e2*(vf_tmp*vf_tmp);
				dtef_n *= vf_tmp;

				vf_tmp = 1.0f/(a_n+c_n*(1.0f-ee_vf[k-1])
								 -(dti2*dtef_n+1.0f));

				ee_vf[k] = a_n*vf_tmp;

				gg_vf[k] = (dti2*(-prod_n*l_n*e1)
						+c_n*gg_vf[k-1]
					    -vf[k_off+j_off+i])*vf_tmp;
			}
		}

		/////////////////////////////
		////k=0, k=kb-1
		//for (k = 0; k < kb; k++){

		//	sh[k] = coef1/(1.0f-coef2*gh[k]);
		//	sm[k] = coef3+sh[k]*coef4*gh[k];
		//	sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

		//	prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
		//}
		/////////////////////////////


		kq[kb_2_off+j_off+i] = (prodk_n*0.41f*sh_n
							   +kq[kb_2_off+j_off+i])*0.5f;

		kq[kb_1_off+j_off+i] = (prodk_bot*0.41f*sh_bot
							   +kq[kb_1_off+j_off+i])*0.5f;

		km[j_off+i] = (prodk_surf*sm_surf
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prodk_bot*sm_bot
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prodk_surf*sh_surf
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prodk_bot*sh_bot
							   +kh[kb_1_off+j_off+i])*0.5f;

		//////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////

		for (ki = kb-2; ki >= 0; ki--){
			uf[ki_off+j_off+i] = ee_uf[ki]*uf[ki_A1_off+j_off+i]
	   					 	+gg_uf[ki];	
			uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		}


		uf[j_off+i] *= fsm[j_off+i];


		for (ki = kb-2; ki > 0; ki--){
			vf[ki_off+j_off+i] = ee_vf[ki]*vf[ki_A1_off+j_off+i]
								+gg_vf[ki];	

			vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		}
		vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i])*fsm[j_off+i];
	
	
	
	}
	
}

__global__ void
profq_overlap_bcond_ew_gpu_kernel_1(const float * __restrict__ t, 
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
				   float * __restrict__ kq, 
				   float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ wubot, 
				   const float * __restrict__ wvbot,
				   const float * __restrict__ h, 
				   const float * __restrict__ z, 
				   const float * __restrict__ zz, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   float umol, 
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y+1; 
	int i;

	if (blockIdx.x == 0){
		i = threadIdx.x+1;	
	}else{
		i = im-2-threadIdx.x;	
	}

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

    float a1,a2,b1,b2,c1;
    float coef1,coef2,coef3,coef4,coef5;
    float e1,e2;
    float cbcnst, surfl, shiw;
    float p,sef,sp,tp;

    a1=0.92f;b1=16.6f;a2=0.74f;b2=10.1f;c1=0.08f;
    e1=1.8e0f;e2=1.33e0f;
    cbcnst=100.0f;surfl=2.0e5f;shiw=0.0f;
    sef=1.e0f;

	float const1;
    const1=powf(16.6e0f,(2.e0f/3.e0f))*sef;
	
	float ee_uf[k_size], ee_vf[k_size];
    float gg_uf[k_size], gg_vf[k_size];
    float stf;	
	
//	float a[k_size];
//	float c[k_size];
//	float ee[k_size];
//	float gg[k_size];
//	float sm[k_size];
//	float sh[k_size];
//	float cc[k_size];
//	float gh[k_size];
//	float boygr[k_size];
//	float stf;
//	float prod_0[k_size];
//	float prod[k_size];
//	float dtef[k_size];
//	float l[k_size];

	float dh, l0, utau2;
/*
	if (j < jm-1){
		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];

		ee[0] = 0;
		gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		for (k = 0; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc[k] = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));
		}


		l[0] = kappa*l0;	
		l[kb-1] = 0;
		gh[0] = 0;
		gh[kb-1] = 0;


		stf = 1.0f;
		for (k = 1; k < kbm1; k++){
			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr[k] = grav*(rho[k_1_off+j_off+i]
							-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc[k-1]*cc[k-1])
						 +(cc[k]*cc[k]));

			l[k] = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l[k] = MAX(l[k], (kappa*l0));

			gh[k] = (l[k]*l[k])*boygr[k]
				   /(q2b[k_off+j_off+i]+small);

			gh[k] = MIN(gh[k], 0.028f);

			dtef[k] = sqrtf(ABS(q2b[k_off+j_off+i]))
						   *stf/(b1*l[k]+small);
		}

		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		for (k = 0; k < kb; k++){

			sh[k] = coef1/(1.0f-coef2*gh[k]);
			sm[k] = coef3+sh[k]*coef4*gh[k];
			sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

			prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
		}

		for (k = 1; k < kbm1; k++){
			a[k] = -dti2*(kq[k_A1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k]*dh*dh);

			c[k] = -dti2*(kq[k_1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k-1]*dh*dh);
			prod[k] = 0;

			kq[k_1_off+j_off+i] = (prod_0[k-1]*0.41f*sh[k-1]
								+kq[k_1_off+j_off+i])*0.5f;
		}

		kq[kb_2_off+j_off+i] = (prod_0[kb-2]*0.41f*sh[kb-2]
							   +kq[kb_2_off+j_off+i])*0.5f;
		kq[kb_1_off+j_off+i] = (prod_0[kb-1]*0.41f*sh[kb-1]
							   +kq[kb_1_off+j_off+i])*0.5f;


		for (k = 1; k < kbm1; k++){
			float tmpu = u[k_off+j_off+i]
						-u[k_1_off+j_off+i]
						+u[k_off+j_off+(i+1)]
						-u[k_1_off+j_off+(i+1)];

			float tmpv = v[k_off+j_off+i]
						-v[k_1_off+j_off+i]
						+v[k_off+j_A1_off+i]
						-v[k_1_off+j_A1_off+i];

			prod[k] = km[k_off+j_off+i]*0.25f*sef
					    *((tmpu*tmpu)+(tmpv*tmpv))
				        /((dzz[k-1]*dh)
					      *(dzz[k-1]*dh))
					 -shiw*km[k_off+j_off+i]
					    *boygr[k];

			prod[k] = prod[k]+kh[k_off+j_off+i]*boygr[k];

			km[k_off+j_off+i] = (prod_0[k]*sm[k]
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prod_0[k]*sh[k]
								+kh[k_off+j_off+i])*0.5f;
		}

		km[j_off+i] = (prod_0[0]*sm[0]
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prod_0[kb-1]*sm[kb-1]
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prod_0[0]*sh[0]
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prod_0[kb-1]*sh[kb-1]
							   +kh[kb_1_off+j_off+i])*0.5f;

		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
									+(tmpv_bot*tmpv_bot))
							  *const1;

		for (k = 1; k < kbm1; k++){
			float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
	   						 -(2.0f*dti2*dtef[k]+1.0f));

	   		ee[k] = a[k]*tmp;

	   		gg[k] = (-2.0f*dti2*prod[k]
	   				 +c[k]*gg[k-1]
	   				 -uf[k_off+j_off+i])*tmp;
		}

		for (ki = kb-2; ki >= 0; ki--){
			uf[ki_off+j_off+i] = ee[ki]*uf[ki_A1_off+j_off+i]
	   					 	+gg[ki];	
			uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		}

		vf[j_off+i] = 0;
		vf[kb_1_off+j_off+i] = 0;
		ee[1] = 0;
		gg[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
									*dh*q2[kbm1_1_off+j_off+i];

		for (k = 1; k < kbm1; k++){
			float tmp = (1.0f/ABS(z[k]-z[0])
							+1.0f/ABS(z[k]-z[kb-1]))
						*l[k]/(dh*kappa);

			tmp = 1.0f+e2*(tmp*tmp);
			dtef[k] *= tmp;
		}

		for (k = 2; k < kbm1; k++){
			float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
							 -(dti2*dtef[k]+1.0f));

			ee[k] = a[k]*tmp;

			gg[k] = (dti2*(-prod[k]*l[k]*e1)
					+c[k]*gg[k-1]
				    -vf[k_off+j_off+i])*tmp;
		}

		for (ki = kb-2; ki > 0; ki--){
			vf[ki_off+j_off+i] = ee[ki]*vf[ki_A1_off+j_off+i]
								+gg[ki];	

			vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		}
		vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i]);

	}
*/
				   
		
	
	
	
	
	
	
	
	
	
	
	if (j < jm-1){

		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];


		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		//l[0] = kappa*l0;	
		//l[kb-1] = 0;
		//gh[0] = 0;
		//gh[kb-1] = 0;
		stf = 1.0f;

		float cc_b, cc_n;
		float cc_tmp;
		tp = t[j_off+i] + tbias;
		sp = s[j_off+i] + sbias;
		p = grav*rhoref*(-zz[0]*h[j_off+i])*1.0e-4f;
		cc_tmp = 1449.1f+0.00821f*p
			   +4.55f*tp-0.045f*(tp*tp)
			   +1.34f*(sp-35.0f);

		cc_b = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
						   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

		float gh_surf = 0.0f, gh_bot = 0; 
		float gh_n;
		float l_surf = kappa*l0, l_bot = 0, l_n;
		float sh_surf = 0, sh_bot = 0; 
		float sh_b, sh_n;
		float sm_surf = 0, sm_bot = 0, sm_n;
		float prodk_surf = 0, prodk_bot = 0, prodk_b, prodk_n;
		float boygr_n, a_n, c_n, dtef_n, prod_n;

		ee_uf[0] = 0;
		gg_uf[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;

		vf[j_off+i] = 0;
		vf[kb_1_off+j_off+i] = 0;
		ee_vf[1] = 0;
		gg_vf[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
									*dh*q2[kbm1_1_off+j_off+i];


		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		sh_surf = coef1/(1.0f-coef2*gh_surf);
		sh_bot = coef1/(1.0f-coef2*gh_bot);
		sh_b = sh_surf;

		sm_surf = coef3+sh_surf*coef4*gh_surf;
		sm_surf = sm_surf/(1.0f-coef5*gh_surf); 
		sm_bot = coef3+sh_bot*coef4*gh_bot;
		sm_bot = sm_bot/(1.0f-coef5*gh_bot); 

		prodk_surf = l_surf*sqrtf(ABS(q2[j_off+i]));	
		prodk_bot = l_bot*sqrtf(ABS(q2[kb_1_off+j_off+i]));	
		prodk_b = prodk_surf;
		//prodk_b = prod_0[0];

		for (k = 1; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc_n = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr_n = grav*(rho[k_1_off+j_off+i]
						-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc_b*cc_b)
						 +(cc_n*cc_n));

			cc_b = cc_n;

			l_n = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l_n = MAX(l_n, (kappa*l0));

			gh_n = (l_n*l_n)*boygr_n
				   /(q2b[k_off+j_off+i]+small);

			gh_n = MIN(gh_n, 0.028f);

			dtef_n = sqrtf(ABS(q2b[k_off+j_off+i]))
						   *stf/(b1*l_n+small);

			sh_n = coef1/(1.0f-coef2*gh_n);
			sm_n = coef3+sh_n*coef4*gh_n;
			sm_n = sm_n/(1.0f-coef5*gh_n); 

			prodk_n= l_n*sqrtf(ABS(q2[k_off+j_off+i]));	

			a_n = -dti2*(kq[k_A1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k]*dh*dh);

			c_n = -dti2*(kq[k_1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k-1]*dh*dh);
			//prod[k] = 0;

			kq[k_1_off+j_off+i] = (prodk_b*0.41f*sh_b
								  +kq[k_1_off+j_off+i])*0.5f;

			float tmpu = u[k_off+j_off+i]
						-u[k_1_off+j_off+i]
						+u[k_off+j_off+(i+1)]
						-u[k_1_off+j_off+(i+1)];

			float tmpv = v[k_off+j_off+i]
						-v[k_1_off+j_off+i]
						+v[k_off+j_A1_off+i]
						-v[k_1_off+j_A1_off+i];

			prod_n = km[k_off+j_off+i]*0.25f*sef
					    *((tmpu*tmpu)+(tmpv*tmpv))
				        /((dzz[k-1]*dh)
					      *(dzz[k-1]*dh))
					 -shiw*km[k_off+j_off+i]
					    *boygr_n;

			prod_n = prod_n+kh[k_off+j_off+i]*boygr_n;

			km[k_off+j_off+i] = (prodk_n*sm_n
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prodk_n*sh_n
								+kh[k_off+j_off+i])*0.5f;

			sh_b = sh_n;
			prodk_b = prodk_n;
	    
		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
		
		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
									+(tmpv_bot*tmpv_bot))
							  *const1;


			/////////////////////////////
			/////////////////////////////

			float uf_tmp = 1.0f/(a_n+c_n*(1.0f-ee_uf[k-1])
	   						 -(2.0f*dti2*dtef_n+1.0f));

	   		ee_uf[k] = a_n*uf_tmp;

	   		gg_uf[k] = (-2.0f*dti2*prod_n
	   				 +c_n*gg_uf[k-1]
	   				 -uf[k_off+j_off+i])*uf_tmp;

			/////////////////////////////
			/////////////////////////////

			if (k > 1){
				float vf_tmp = (1.0f/ABS(z[k]-z[0])
								+1.0f/ABS(z[k]-z[kb-1]))
							*l_n/(dh*kappa);

				vf_tmp = 1.0f+e2*(vf_tmp*vf_tmp);
				dtef_n *= vf_tmp;

				vf_tmp = 1.0f/(a_n+c_n*(1.0f-ee_vf[k-1])
								 -(dti2*dtef_n+1.0f));

				ee_vf[k] = a_n*vf_tmp;

				gg_vf[k] = (dti2*(-prod_n*l_n*e1)
						+c_n*gg_vf[k-1]
					    -vf[k_off+j_off+i])*vf_tmp;
			}
		}

		/////////////////////////////
		////k=0, k=kb-1
		//for (k = 0; k < kb; k++){

		//	sh[k] = coef1/(1.0f-coef2*gh[k]);
		//	sm[k] = coef3+sh[k]*coef4*gh[k];
		//	sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

		//	prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
		//}
		/////////////////////////////


		kq[kb_2_off+j_off+i] = (prodk_n*0.41f*sh_n
							   +kq[kb_2_off+j_off+i])*0.5f;

		kq[kb_1_off+j_off+i] = (prodk_bot*0.41f*sh_bot
							   +kq[kb_1_off+j_off+i])*0.5f;

		km[j_off+i] = (prodk_surf*sm_surf
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prodk_bot*sm_bot
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prodk_surf*sh_surf
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prodk_bot*sh_bot
							   +kh[kb_1_off+j_off+i])*0.5f;

		//////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////

		for (ki = kb-2; ki >= 0; ki--){
			uf[ki_off+j_off+i] = ee_uf[ki]*uf[ki_A1_off+j_off+i]
	   					 	+gg_uf[ki];	
			uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		}

		for (ki = kb-2; ki > 0; ki--){
			vf[ki_off+j_off+i] = ee_vf[ki]*vf[ki_A1_off+j_off+i]
								+gg_vf[ki];	

			vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		}
		vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i]);
	}
		   
				   
				   








}

__global__ void
profq_overlap_bcond_sn_gpu_kernel_1(const float * __restrict__ t, 
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
				   float * __restrict__ kq, 
				   float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ wubot, 
				   const float * __restrict__ wvbot,
				   const float * __restrict__ h, 
				   const float * __restrict__ z, 
				   const float * __restrict__ zz, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   float umol, 
				   int kb, int jm, int im){

	int k, ki;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y < 8){
		j = blockIdx.y*blockDim.y+threadIdx.y+1;	
	}else{
		j = jm-2-((blockIdx.y-8)*blockDim.y+threadIdx.y);
	}

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

    float a1,a2,b1,b2,c1;
    float coef1,coef2,coef3,coef4,coef5;
    float e1,e2;
    float cbcnst, surfl, shiw;
    float p,sef,sp,tp;

    a1=0.92f;b1=16.6f;a2=0.74f;b2=10.1f;c1=0.08f;
    e1=1.8e0f;e2=1.33e0f;
    cbcnst=100.0f;surfl=2.0e5f;shiw=0.0f;
    sef=1.e0f;

	float const1;
    const1=powf(16.6e0f,(2.e0f/3.e0f))*sef;
    
	float ee_uf[k_size], ee_vf[k_size];
    float gg_uf[k_size], gg_vf[k_size];
    float stf;	
	

	//float a[k_size];
	//float c[k_size];
	//float ee[k_size];
	//float gg[k_size];
	//float sm[k_size];
	//float sh[k_size];
	//float cc[k_size];
	//float gh[k_size];
	//float boygr[k_size];
	//float stf;
	//float prod_0[k_size];
	//float prod[k_size];
	//float dtef[k_size];
	//float l[k_size];

	float dh, l0, utau2;
/*
	if (i > 32 && i < im-33){ 
		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];

		ee[0] = 0;
		gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		for (k = 0; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc[k] = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));
		}


		l[0] = kappa*l0;	
		l[kb-1] = 0;
		gh[0] = 0;
		gh[kb-1] = 0;


		stf = 1.0f;
		for (k = 1; k < kbm1; k++){
			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr[k] = grav*(rho[k_1_off+j_off+i]
							-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc[k-1]*cc[k-1])
						 +(cc[k]*cc[k]));

			l[k] = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l[k] = MAX(l[k], (kappa*l0));

			gh[k] = (l[k]*l[k])*boygr[k]
				   /(q2b[k_off+j_off+i]+small);

			gh[k] = MIN(gh[k], 0.028f);

			dtef[k] = sqrtf(ABS(q2b[k_off+j_off+i]))
						   *stf/(b1*l[k]+small);
		}

		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		for (k = 0; k < kb; k++){

			sh[k] = coef1/(1.0f-coef2*gh[k]);
			sm[k] = coef3+sh[k]*coef4*gh[k];
			sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

			prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
		}

		for (k = 1; k < kbm1; k++){
			a[k] = -dti2*(kq[k_A1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k]*dh*dh);

			c[k] = -dti2*(kq[k_1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k-1]*dh*dh);
			prod[k] = 0;

			kq[k_1_off+j_off+i] = (prod_0[k-1]*0.41f*sh[k-1]
								+kq[k_1_off+j_off+i])*0.5f;
		}

		kq[kb_2_off+j_off+i] = (prod_0[kb-2]*0.41f*sh[kb-2]
							   +kq[kb_2_off+j_off+i])*0.5f;
		kq[kb_1_off+j_off+i] = (prod_0[kb-1]*0.41f*sh[kb-1]
							   +kq[kb_1_off+j_off+i])*0.5f;


		for (k = 1; k < kbm1; k++){
			float tmpu = u[k_off+j_off+i]
						-u[k_1_off+j_off+i]
						+u[k_off+j_off+(i+1)]
						-u[k_1_off+j_off+(i+1)];

			float tmpv = v[k_off+j_off+i]
						-v[k_1_off+j_off+i]
						+v[k_off+j_A1_off+i]
						-v[k_1_off+j_A1_off+i];

			prod[k] = km[k_off+j_off+i]*0.25f*sef
					    *((tmpu*tmpu)+(tmpv*tmpv))
				        /((dzz[k-1]*dh)
					      *(dzz[k-1]*dh))
					 -shiw*km[k_off+j_off+i]
					    *boygr[k];

			prod[k] = prod[k]+kh[k_off+j_off+i]*boygr[k];

			km[k_off+j_off+i] = (prod_0[k]*sm[k]
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prod_0[k]*sh[k]
								+kh[k_off+j_off+i])*0.5f;
		}

		km[j_off+i] = (prod_0[0]*sm[0]
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prod_0[kb-1]*sm[kb-1]
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prod_0[0]*sh[0]
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prod_0[kb-1]*sh[kb-1]
							   +kh[kb_1_off+j_off+i])*0.5f;

		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
									+(tmpv_bot*tmpv_bot))
							  *const1;

		for (k = 1; k < kbm1; k++){
			float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
	   						 -(2.0f*dti2*dtef[k]+1.0f));

	   		ee[k] = a[k]*tmp;

	   		gg[k] = (-2.0f*dti2*prod[k]
	   				 +c[k]*gg[k-1]
	   				 -uf[k_off+j_off+i])*tmp;
		}

		for (ki = kb-2; ki >= 0; ki--){
			uf[ki_off+j_off+i] = ee[ki]*uf[ki_A1_off+j_off+i]
	   					 	+gg[ki];	
			uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		}

		vf[j_off+i] = 0;
		vf[kb_1_off+j_off+i] = 0;
		ee[1] = 0;
		gg[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
									*dh*q2[kbm1_1_off+j_off+i];

		for (k = 1; k < kbm1; k++){
			float tmp = (1.0f/ABS(z[k]-z[0])
							+1.0f/ABS(z[k]-z[kb-1]))
						*l[k]/(dh*kappa);

			tmp = 1.0f+e2*(tmp*tmp);
			dtef[k] *= tmp;
		}

		for (k = 2; k < kbm1; k++){
			float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
							 -(dti2*dtef[k]+1.0f));

			ee[k] = a[k]*tmp;

			gg[k] = (dti2*(-prod[k]*l[k]*e1)
					+c[k]*gg[k-1]
				    -vf[k_off+j_off+i])*tmp;
		}

		for (ki = kb-2; ki > 0; ki--){
			vf[ki_off+j_off+i] = ee[ki]*vf[ki_A1_off+j_off+i]
								+gg[ki];	

			vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		}
		vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i]);


	}
*/
	
	if (i > 32 && i < im-33){

		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];


		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		//l[0] = kappa*l0;	
		//l[kb-1] = 0;
		//gh[0] = 0;
		//gh[kb-1] = 0;
		stf = 1.0f;

		float cc_b, cc_n;
		float cc_tmp;
		tp = t[j_off+i] + tbias;
		sp = s[j_off+i] + sbias;
		p = grav*rhoref*(-zz[0]*h[j_off+i])*1.0e-4f;
		cc_tmp = 1449.1f+0.00821f*p
			   +4.55f*tp-0.045f*(tp*tp)
			   +1.34f*(sp-35.0f);

		cc_b = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
						   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

		float gh_surf = 0.0f, gh_bot = 0; 
		float gh_n;
		float l_surf = kappa*l0, l_bot = 0, l_n;
		float sh_surf = 0, sh_bot = 0; 
		float sh_b, sh_n;
		float sm_surf = 0, sm_bot = 0, sm_n;
		float prodk_surf = 0, prodk_bot = 0, prodk_b, prodk_n;
		float boygr_n, a_n, c_n, dtef_n, prod_n;

		ee_uf[0] = 0;
		gg_uf[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;

		vf[j_off+i] = 0;
		vf[kb_1_off+j_off+i] = 0;
		ee_vf[1] = 0;
		gg_vf[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
									*dh*q2[kbm1_1_off+j_off+i];


		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		sh_surf = coef1/(1.0f-coef2*gh_surf);
		sh_bot = coef1/(1.0f-coef2*gh_bot);
		sh_b = sh_surf;

		sm_surf = coef3+sh_surf*coef4*gh_surf;
		sm_surf = sm_surf/(1.0f-coef5*gh_surf); 
		sm_bot = coef3+sh_bot*coef4*gh_bot;
		sm_bot = sm_bot/(1.0f-coef5*gh_bot); 

		prodk_surf = l_surf*sqrtf(ABS(q2[j_off+i]));	
		prodk_bot = l_bot*sqrtf(ABS(q2[kb_1_off+j_off+i]));	
		prodk_b = prodk_surf;
		//prodk_b = prod_0[0];

		for (k = 1; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc_n = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr_n = grav*(rho[k_1_off+j_off+i]

						-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc_b*cc_b)
						 +(cc_n*cc_n));

			cc_b = cc_n;

			l_n = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l_n = MAX(l_n, (kappa*l0));

			gh_n = (l_n*l_n)*boygr_n
				   /(q2b[k_off+j_off+i]+small);

			gh_n = MIN(gh_n, 0.028f);

			dtef_n = sqrtf(ABS(q2b[k_off+j_off+i]))
						   *stf/(b1*l_n+small);

			sh_n = coef1/(1.0f-coef2*gh_n);
			sm_n = coef3+sh_n*coef4*gh_n;
			sm_n = sm_n/(1.0f-coef5*gh_n); 

			prodk_n= l_n*sqrtf(ABS(q2[k_off+j_off+i]));	

			a_n = -dti2*(kq[k_A1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k]*dh*dh);

			c_n = -dti2*(kq[k_1_off+j_off+i]
						  +kq[k_off+j_off+i]
						  +2.0f*umol)
						*0.5f
						/(dzz[k-1]*dz[k-1]*dh*dh);
			//prod[k] = 0;

			kq[k_1_off+j_off+i] = (prodk_b*0.41f*sh_b
								  +kq[k_1_off+j_off+i])*0.5f;

			float tmpu = u[k_off+j_off+i]
						-u[k_1_off+j_off+i]
						+u[k_off+j_off+(i+1)]
						-u[k_1_off+j_off+(i+1)];

			float tmpv = v[k_off+j_off+i]
						-v[k_1_off+j_off+i]
						+v[k_off+j_A1_off+i]
						-v[k_1_off+j_A1_off+i];

			prod_n = km[k_off+j_off+i]*0.25f*sef
					    *((tmpu*tmpu)+(tmpv*tmpv))
				        /((dzz[k-1]*dh)
					      *(dzz[k-1]*dh))
					 -shiw*km[k_off+j_off+i]
					    *boygr_n;

			prod_n = prod_n+kh[k_off+j_off+i]*boygr_n;

			km[k_off+j_off+i] = (prodk_n*sm_n
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prodk_n*sh_n
								+kh[k_off+j_off+i])*0.5f;

			sh_b = sh_n;
			prodk_b = prodk_n;
	
		float tmpu_bot = 0.5f*(wubot[j_off+i]+wubot[j_off+(i+1)]);
		float tmpv_bot = 0.5f*(wvbot[j_off+i]+wvbot[j_A1_off+i]);
		uf[kb_1_off+j_off+i] = sqrtf((tmpu_bot*tmpu_bot)
									+(tmpv_bot*tmpv_bot))
							  *const1;


			/////////////////////////////
			/////////////////////////////

			float uf_tmp = 1.0f/(a_n+c_n*(1.0f-ee_uf[k-1])
	   						 -(2.0f*dti2*dtef_n+1.0f));

	   		ee_uf[k] = a_n*uf_tmp;

	   		gg_uf[k] = (-2.0f*dti2*prod_n
	   				 +c_n*gg_uf[k-1]
	   				 -uf[k_off+j_off+i])*uf_tmp;

			/////////////////////////////
			/////////////////////////////

			if (k > 1){
				float vf_tmp = (1.0f/ABS(z[k]-z[0])
								+1.0f/ABS(z[k]-z[kb-1]))
							*l_n/(dh*kappa);

				vf_tmp = 1.0f+e2*(vf_tmp*vf_tmp);
				dtef_n *= vf_tmp;

				vf_tmp = 1.0f/(a_n+c_n*(1.0f-ee_vf[k-1])
								 -(dti2*dtef_n+1.0f));

				ee_vf[k] = a_n*vf_tmp;

				gg_vf[k] = (dti2*(-prod_n*l_n*e1)
						+c_n*gg_vf[k-1]
					    -vf[k_off+j_off+i])*vf_tmp;
			}
		}

		/////////////////////////////
		////k=0, k=kb-1
		//for (k = 0; k < kb; k++){

		//	sh[k] = coef1/(1.0f-coef2*gh[k]);
		//	sm[k] = coef3+sh[k]*coef4*gh[k];
		//	sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

		//	prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
		//}
		/////////////////////////////


		kq[kb_2_off+j_off+i] = (prodk_n*0.41f*sh_n
							   +kq[kb_2_off+j_off+i])*0.5f;

		kq[kb_1_off+j_off+i] = (prodk_bot*0.41f*sh_bot
							   +kq[kb_1_off+j_off+i])*0.5f;

		km[j_off+i] = (prodk_surf*sm_surf
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prodk_bot*sm_bot
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prodk_surf*sh_surf
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prodk_bot*sh_bot
							   +kh[kb_1_off+j_off+i])*0.5f;

		//////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////

		for (ki = kb-2; ki >= 0; ki--){
			uf[ki_off+j_off+i] = ee_uf[ki]*uf[ki_A1_off+j_off+i]
	   					 	+gg_uf[ki];	
			uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		}

		for (ki = kb-2; ki > 0; ki--){
			vf[ki_off+j_off+i] = ee_vf[ki]*vf[ki_A1_off+j_off+i]
								+gg_vf[ki];	

			vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		}
		vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i]);
	}
}

__global__ void
profq_overlap_bcond_ew_bcond_gpu_kernel_1(
				   const float * __restrict__ t, 
				   const float * __restrict__ s,
				   //const float * __restrict__ u, 
				   //const float * __restrict__ v,
				   //float * __restrict__ uf, 
				   //float * __restrict__ vf,
				   float * __restrict__ q2b, 
				   float * __restrict__ q2lb,
				   const float * __restrict__ q2,
				   const float * __restrict__ rho,
				   const float * __restrict__ etf, 
				   const float * __restrict__ cu_utau2,
				   float * __restrict__ kq, 
				   float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ h, 
				   const float * __restrict__ z, 
				   const float * __restrict__ zz, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   float umol, 
				   int kb, int jm, int im){

	int k, ki;
	const int j = blockDim.y*blockIdx.y + threadIdx.y; 
	int i;

	if (blockIdx.x == 0){
		i = 0;	
	}else{
		i = im-1;	
	}

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

    float a1,a2,b1,b2,c1;
    float coef1,coef2,coef3,coef4,coef5;
    float e1,e2;
    float cbcnst, surfl, shiw;
    float p,sef,sp,tp;

    a1=0.92f;b1=16.6f;a2=0.74f;b2=10.1f;c1=0.08f;
    e1=1.8e0f;e2=1.33e0f;
    cbcnst=100.0f;surfl=2.0e5f;shiw=0.0f;
    sef=1.e0f;

	float stf;

/*
	//float a[k_size];
	//float c[k_size];
	//float ee[k_size];
	//float gg[k_size];
	float sm[k_size];
	float sh[k_size];
	float cc[k_size];
	float gh[k_size];
	float boygr[k_size];
	float stf;
	float prod_0[k_size];
	//float prod[k_size];
	//float dtef[k_size];
	float l[k_size];
*/
	float dh, l0, utau2;
/*
	if (j < jm){
		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];

		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		for (k = 0; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc[k] = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));
		}


		l[0] = kappa*l0;	
		l[kb-1] = 0;
		gh[0] = 0;
		gh[kb-1] = 0;

		stf = 1.0f;
		for (k = 1; k < kbm1; k++){
			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr[k] = grav*(rho[k_1_off+j_off+i]
							-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc[k-1]*cc[k-1])
						 +(cc[k]*cc[k]));

			l[k] = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l[k] = MAX(l[k], (kappa*l0));

			gh[k] = (l[k]*l[k])*boygr[k]
				   /(q2b[k_off+j_off+i]+small);

			gh[k] = MIN(gh[k], 0.028f);

			//dtef[k] = sqrtf(ABS(q2b[k_off+j_off+i]))
			//			   *stf/(b1*l[k]+small);
		}

		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		for (k = 0; k < kb; k++){

			sh[k] = coef1/(1.0f-coef2*gh[k]);
			sm[k] = coef3+sh[k]*coef4*gh[k];
			sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

			prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	

			kq[k_off+j_off+i] = (prod_0[k]*0.41f*sh[k]
								+kq[k_off+j_off+i])*0.5f;

			km[k_off+j_off+i] = (prod_0[k]*sm[k]
								+km[k_off+j_off+i])*0.5f;

			kh[k_off+j_off+i] = (prod_0[k]*sh[k]
								+kh[k_off+j_off+i])*0.5f;
		}

*/



	if (j < jm){

		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];


		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		//l[0] = kappa*l0;	
		//l[kb-1] = 0;
		//gh[0] = 0;
		//gh[kb-1] = 0;
		stf = 1.0f;

		float cc_b, cc_n;
		float cc_tmp;
		tp = t[j_off+i] + tbias;
		sp = s[j_off+i] + sbias;
		p = grav*rhoref*(-zz[0]*h[j_off+i])*1.0e-4f;
		cc_tmp = 1449.1f+0.00821f*p
			   +4.55f*tp-0.045f*(tp*tp)
			   +1.34f*(sp-35.0f);

		cc_b = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
						   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

		float gh_surf = 0.0f, gh_bot = 0; 
		float gh_n;
		float l_surf = kappa*l0, l_bot = 0, l_n;
		float sh_surf = 0, sh_bot = 0; 
		float sh_b, sh_n;
		float sm_surf = 0, sm_bot = 0, sm_n;
		float prodk_surf = 0, prodk_bot = 0, prodk_b, prodk_n;
		float boygr_n, a_n, c_n, dtef_n, prod_n;


		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		sh_surf = coef1/(1.0f-coef2*gh_surf);
		sh_bot = coef1/(1.0f-coef2*gh_bot);
		sh_b = sh_surf;

		sm_surf = coef3+sh_surf*coef4*gh_surf;
		sm_surf = sm_surf/(1.0f-coef5*gh_surf); 
		sm_bot = coef3+sh_bot*coef4*gh_bot;
		sm_bot = sm_bot/(1.0f-coef5*gh_bot); 

		prodk_surf = l_surf*sqrtf(ABS(q2[j_off+i]));	
		prodk_bot = l_bot*sqrtf(ABS(q2[kb_1_off+j_off+i]));	
		prodk_b = prodk_surf;
		//prodk_b = prod_0[0];

		for (k = 1; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc_n = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr_n = grav*(rho[k_1_off+j_off+i]

						-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc_b*cc_b)
						 +(cc_n*cc_n));

            cc_b = cc_n;
			
			l_n = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l_n = MAX(l_n, (kappa*l0));

			gh_n = (l_n*l_n)*boygr_n
				   /(q2b[k_off+j_off+i]+small);

			gh_n = MIN(gh_n, 0.028f);

			sh_n = coef1/(1.0f-coef2*gh_n);
			sm_n = coef3+sh_n*coef4*gh_n;
			sm_n = sm_n/(1.0f-coef5*gh_n); 

			prodk_n= l_n*sqrtf(ABS(q2[k_off+j_off+i]));	
            
			kq[k_1_off+j_off+i] = (prodk_b*0.41f*sh_b
					              +kq[k_1_off+j_off+i])*0.5f;

			km[k_off+j_off+i] = (prodk_n*sm_n
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prodk_n*sh_n
								+kh[k_off+j_off+i])*0.5f;
            sh_b = sh_n;
			prodk_b = prodk_n;

		kq[kb_2_off+j_off+i] = (prodk_n*0.41f*sh_n
							   +kq[kb_2_off+j_off+i])*0.5f;

		kq[kb_1_off+j_off+i] = (prodk_bot*0.41f*sh_bot
							   +kq[kb_1_off+j_off+i])*0.5f;

		km[j_off+i] = (prodk_surf*sm_surf
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prodk_bot*sm_bot
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prodk_surf*sh_surf
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prodk_bot*sh_bot
							   +kh[kb_1_off+j_off+i])*0.5f;
		}









		//for (k = 1; k < kbm1; k++){
		//	float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
	   	//					 -(2.0f*dti2*dtef[k]+1.0f));

	   	//	ee[k] = a[k]*tmp;

	   	//	gg[k] = (-2.0f*dti2*prod[k]
	   	//			 +c[k]*gg[k-1]
	   	//			 -uf[k_off+j_off+i])*tmp;
		//}

		//for (ki = kb-2; ki >= 0; ki--){
		//	uf[ki_off+j_off+i] = ee[ki]*uf[ki_A1_off+j_off+i]
	   	//				 	+gg[ki];	
		//	uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		//}

		//vf[j_off+i] = 0;
		//vf[kb_1_off+j_off+i] = 0;
		//ee[1] = 0;
		//gg[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		//vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
		//							*dh*q2[kbm1_1_off+j_off+i];

		//for (k = 1; k < kbm1; k++){
		//	float tmp = (1.0f/ABS(z[k]-z[0])
		//					+1.0f/ABS(z[k]-z[kb-1]))
		//				*l[k]/(dh*kappa);

		//	tmp = 1.0f+e2*(tmp*tmp);
		//	dtef[k] *= tmp;
		//}

		//for (k = 2; k < kbm1; k++){
		//	float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
		//					 -(dti2*dtef[k]+1.0f));

		//	ee[k] = a[k]*tmp;

		//	gg[k] = (dti2*(-prod[k]*l[k]*e1)
		//			+c[k]*gg[k-1]
		//		    -vf[k_off+j_off+i])*tmp;
		//}

		//for (ki = kb-2; ki > 0; ki--){
		//	vf[ki_off+j_off+i] = ee[ki]*vf[ki_A1_off+j_off+i]
		//						+gg[ki];	

		//	vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		//}
		//vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i]);

	}
}

__global__ void
profq_overlap_bcond_sn_bcond_gpu_kernel_1(
				   const float * __restrict__ t, 
				   const float * __restrict__ s,
				   //const float * __restrict__ u, 
				   //const float * __restrict__ v,
				   //float * __restrict__ uf, 
				   //float * __restrict__ vf,
				   float * __restrict__ q2b, 
				   float * __restrict__ q2lb,
				   const float * __restrict__ q2,
				   const float * __restrict__ rho,
				   const float * __restrict__ etf, 
				   const float * __restrict__ cu_utau2,
				   float * __restrict__ kq, 
				   float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ h, 
				   const float * __restrict__ z, 
				   const float * __restrict__ zz, 
				   const float * __restrict__ dz, 
				   const float * __restrict__ dzz,
				   float grav, float rhoref, float kappa,
				   float tbias, float sbias,
				   float dti2, float small,
				   float umol, 
				   int kb, int jm, int im){

	int k, ki;
	const int i = blockDim.x*blockIdx.x + threadIdx.x+1; 
	int j;

	if (blockIdx.y == 0){
		j = 0;	
	}else{
		j = jm-1;	
	}

	int kbm1 = kb-1;
	int jmm1 = jm-1;
	int imm1 = im-1;

    float a1,a2,b1,b2,c1;
    float coef1,coef2,coef3,coef4,coef5;
    float e1,e2;
    float cbcnst, surfl, shiw;
    float p,sef,sp,tp;

    a1=0.92f;b1=16.6f;a2=0.74f;b2=10.1f;c1=0.08f;
    e1=1.8e0f;e2=1.33e0f;
    cbcnst=100.0f;surfl=2.0e5f;shiw=0.0f;
    sef=1.e0f;

	float stf;
	
/*
	//float a[k_size];
	//float c[k_size];
	//float ee[k_size];
	//float gg[k_size];
	float sm[k_size];
	float sh[k_size];
	float cc[k_size];
	float gh[k_size];
	float boygr[k_size];
	float stf;
	float prod_0[k_size];
	//float prod[k_size];
	//float dtef[k_size];
	float l[k_size];
*/
	float dh, l0, utau2;
/*
	if (i < im-1){
		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];

		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		for (k = 0; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc[k] = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));
		}


		l[0] = kappa*l0;	
		l[kb-1] = 0;
		gh[0] = 0;
		gh[kb-1] = 0;


		stf = 1.0f;
		for (k = 1; k < kbm1; k++){
			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr[k] = grav*(rho[k_1_off+j_off+i]
							-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc[k-1]*cc[k-1])
						 +(cc[k]*cc[k]));

			l[k] = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l[k] = MAX(l[k], (kappa*l0));

			gh[k] = (l[k]*l[k])*boygr[k]
				   /(q2b[k_off+j_off+i]+small);

			gh[k] = MIN(gh[k], 0.028f);
		}

		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		for (k = 0; k < kb; k++){

			sh[k] = coef1/(1.0f-coef2*gh[k]);
			sm[k] = coef3+sh[k]*coef4*gh[k];
			sm[k]= sm[k]/(1.0f-coef5*gh[k]); 

			prod_0[k]= l[k]*sqrtf(ABS(q2[k_off+j_off+i]));	
			kq[k_off+j_off+i] = (prod_0[k]*0.41f*sh[k]
								+kq[k_off+j_off+i])*0.5f;

			km[k_off+j_off+i] = (prod_0[k]*sm[k]
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prod_0[k]*sh[k]
								+kh[k_off+j_off+i])*0.5f;
		}
*/

	if (i < im-1){

		dh = h[j_off+i]+etf[j_off+i];
		utau2 = cu_utau2[j_off+i];


		//ee[0] = 0;
		//gg[0] = powf((15.8f*cbcnst), (2.f/3.f))*utau2;
		l0 = surfl*utau2/grav;

		//l[0] = kappa*l0;	
		//l[kb-1] = 0;
		//gh[0] = 0;
		//gh[kb-1] = 0;
		stf = 1.0f;

		float cc_b, cc_n;
		float cc_tmp;
		tp = t[j_off+i] + tbias;
		sp = s[j_off+i] + sbias;
		p = grav*rhoref*(-zz[0]*h[j_off+i])*1.0e-4f;
		cc_tmp = 1449.1f+0.00821f*p
			   +4.55f*tp-0.045f*(tp*tp)
			   +1.34f*(sp-35.0f);

		cc_b = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
						   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

		float gh_surf = 0.0f, gh_bot = 0; 
		float gh_n;
		float l_surf = kappa*l0, l_bot = 0, l_n;
		float sh_surf = 0, sh_bot = 0; 
		float sh_b, sh_n;
		float sm_surf = 0, sm_bot = 0, sm_n;
		float prodk_surf = 0, prodk_bot = 0, prodk_b, prodk_n;
		float boygr_n, a_n, c_n, dtef_n, prod_n;


		coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
		coef5=9.0e0f*a1*a2;

		coef1=a2*(1.0f-6.0f*a1/b1*stf);
		coef2=3.0f*a2*b2/stf+18.0f*a1*a2;
		coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf);

		sh_surf = coef1/(1.0f-coef2*gh_surf);
		sh_bot = coef1/(1.0f-coef2*gh_bot);
		sh_b = sh_surf;

		sm_surf = coef3+sh_surf*coef4*gh_surf;
		sm_surf = sm_surf/(1.0f-coef5*gh_surf); 
		sm_bot = coef3+sh_bot*coef4*gh_bot;
		sm_bot = sm_bot/(1.0f-coef5*gh_bot); 

		prodk_surf = l_surf*sqrtf(ABS(q2[j_off+i]));	
		prodk_bot = l_bot*sqrtf(ABS(q2[kb_1_off+j_off+i]));	
		prodk_b = prodk_surf;
		//prodk_b = prod_0[0];

		for (k = 1; k < kbm1; k++){
			float cc_tmp;
			tp = t[k_off+j_off+i] + tbias;
			sp = s[k_off+j_off+i] + sbias;
			p = grav*rhoref*(-zz[k]*h[j_off+i])*1.0e-4f;
			cc_tmp = 1449.1f+0.00821f*p
				   +4.55f*tp-0.045f*(tp*tp)
				   +1.34f*(sp-35.0f);

			cc_n = cc_tmp/sqrtf((1.0f-0.01642f*p/cc_tmp)
							   *(1.0f-0.4f*p/(cc_tmp*cc_tmp)));

			q2b[k_off+j_off+i] = ABS(q2b[k_off+j_off+i]);	
			q2lb[k_off+j_off+i] = ABS(q2lb[k_off+j_off+i]);	

			boygr_n = grav*(rho[k_1_off+j_off+i]

						-rho[k_off+j_off+i])
						  /(dzz[k-1]*h[j_off+i])
					  +(grav*grav)*2.0f
						/((cc_b*cc_b)
						 +(cc_n*cc_n));

            cc_b = cc_n;
			
			l_n = ABS(q2lb[k_off+j_off+i]
					 /(q2b[k_off+j_off+i]+small));
			if (z[k] > -0.5f)
				l_n = MAX(l_n, (kappa*l0));

			gh_n = (l_n*l_n)*boygr_n
				   /(q2b[k_off+j_off+i]+small);

			gh_n = MIN(gh_n, 0.028f);

			sh_n = coef1/(1.0f-coef2*gh_n);
			sm_n = coef3+sh_n*coef4*gh_n;
			sm_n = sm_n/(1.0f-coef5*gh_n); 

			prodk_n= l_n*sqrtf(ABS(q2[k_off+j_off+i]));	
            
			kq[k_1_off+j_off+i] = (prodk_b*0.41f*sh_b
					              +kq[k_1_off+j_off+i])*0.5f;

			km[k_off+j_off+i] = (prodk_n*sm_n
								+km[k_off+j_off+i])*0.5f;
			kh[k_off+j_off+i] = (prodk_n*sh_n
								+kh[k_off+j_off+i])*0.5f;
            sh_b = sh_n;
			prodk_b = prodk_n;

		kq[kb_2_off+j_off+i] = (prodk_n*0.41f*sh_n
							   +kq[kb_2_off+j_off+i])*0.5f;

		kq[kb_1_off+j_off+i] = (prodk_bot*0.41f*sh_bot
							   +kq[kb_1_off+j_off+i])*0.5f;

		km[j_off+i] = (prodk_surf*sm_surf
					  +km[j_off+i])*0.5f;
		km[kb_1_off+j_off+i] = (prodk_bot*sm_bot
							   +km[kb_1_off+j_off+i])*0.5f;

		kh[j_off+i] = (prodk_surf*sh_surf
					  +kh[j_off+i])*0.5f;
		kh[kb_1_off+j_off+i] = (prodk_bot*sh_bot
							   +kh[kb_1_off+j_off+i])*0.5f;
		}






		//for (k = 1; k < kbm1; k++){
		//	float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
	   	//					 -(2.0f*dti2*dtef[k]+1.0f));

	   	//	ee[k] = a[k]*tmp;

	   	//	gg[k] = (-2.0f*dti2*prod[k]
	   	//			 +c[k]*gg[k-1]
	   	//			 -uf[k_off+j_off+i])*tmp;
		//}

		//for (ki = kb-2; ki >= 0; ki--){
		//	uf[ki_off+j_off+i] = ee[ki]*uf[ki_A1_off+j_off+i]
	   	//				 	+gg[ki];	
		//	uf[ki_A1_off+j_off+i] = ABS(uf[ki_A1_off+j_off+i]);	
		//}

		//vf[j_off+i] = 0;
		//vf[kb_1_off+j_off+i] = 0;
		//ee[1] = 0;
		//gg[1] = -kappa*z[1]*dh*q2[1*jm*im+j*im+i];
		//vf[kb_2_off+j_off+i] = kappa*(1.0f+z[kbm1-1])
		//							*dh*q2[kbm1_1_off+j_off+i];

		//for (k = 1; k < kbm1; k++){
		//	float tmp = (1.0f/ABS(z[k]-z[0])
		//					+1.0f/ABS(z[k]-z[kb-1]))
		//				*l[k]/(dh*kappa);

		//	tmp = 1.0f+e2*(tmp*tmp);
		//	dtef[k] *= tmp;
		//}

		//for (k = 2; k < kbm1; k++){
		//	float tmp = 1.0f/(a[k]+c[k]*(1.0f-ee[k-1])
		//					 -(dti2*dtef[k]+1.0f));

		//	ee[k] = a[k]*tmp;

		//	gg[k] = (dti2*(-prod[k]*l[k]*e1)
		//			+c[k]*gg[k-1]
		//		    -vf[k_off+j_off+i])*tmp;
		//}

		//for (ki = kb-2; ki > 0; ki--){
		//	vf[ki_off+j_off+i] = ee[ki]*vf[ki_A1_off+j_off+i]
		//						+gg[ki];	

		//	vf[ki_A1_off+j_off+i] = ABS(vf[ki_A1_off+j_off+i]);
		//}
		//vf[jm*im+j_off+i] = ABS(vf[jm*im+j_off+i]);

	}
}

__global__ void
profq_overlap_bcond_ew_bcond_gpu_kernel_2(float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ fsm,
				   int n_east, int n_west,
				   int kb, int jm, int im){

	int k;
	const int j = blockDim.y*blockIdx.y + threadIdx.y;

	int imm1 = im-1;

	if (n_east == -1){
		if (j < jm){
			for (k = 0; k < kb; k++){
				km[k_off+j_off+(im-1)] = km[k_off+j_off+(imm1-1)]
										*fsm[j_off+im-1];
				kh[k_off+j_off+(im-1)] = kh[k_off+j_off+(imm1-1)]
										*fsm[j_off+im-1];
			}
		}
	}

	if (n_west == -1){
		if (j < jm){
			for (k = 0; k < kb; k++){
				km[k_off+j_off] = km[k_off+j_off+1]
								 *fsm[j_off];
				kh[k_off+j_off] = kh[k_off+j_off+1]
								 *fsm[j_off];
			}
		}
	}
}

__global__ void
profq_overlap_bcond_sn_bcond_gpu_kernel_2(float * __restrict__ km, 
				   float * __restrict__ kh,
				   const float * __restrict__ fsm,
				   int n_north, int n_south, 
				   int kb, int jm, int im){

	int k;
	const int i = blockDim.x*blockIdx.x + threadIdx.x;

	int jmm1 = jm-1;

	if (n_north == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb; k++){
				km[k_off+jm_1_off+i] = km[k_off+jmm1_1_off+i]
									  *fsm[jm_1_off+i];
				kh[k_off+jm_1_off+i] = kh[k_off+jmm1_1_off+i]
									  *fsm[jm_1_off+i];
			}
		}
	}

	if (n_south == -1){
		if (i > 0 && i < im-1){
			for (k = 0; k < kb; k++){
				km[k_off+i] = km[k_off+1*im+i]*fsm[i];	
				kh[k_off+i] = kh[k_off+1*im+i]*fsm[i];
			}
		}
	}
}


void profq_overlap_bcond(){
// modify:
//			uf, vf, 
//	+referencd: kq, km, kh, l //kq is just used in this scope, but not just a local variable 
//	+reference:	q2b, q2lb
	//int i,j,k,ki;
	//comment: dt may not used in this function, because it is in an alternative process
	//comment: after a test, it is confirmed that uf&vf need not copy-in

#ifndef TIME_DISABLE
	struct timeval start_profq,
				   end_profq;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&start_profq);
#endif


	/*
    float ee[k_size][j_size][i_size];
    float gg[k_size][j_size][i_size];
    float l0[j_size][i_size];

	float *d_a = d_3d_tmp0;
	float *d_c = d_3d_tmp1;
	float *d_ee = d_3d_tmp2;
	float *d_gg = d_3d_tmp3;
	float *d_sm = d_3d_tmp4;
	float *d_sh = d_3d_tmp5;
	float *d_cc = d_3d_tmp6;
	float *d_gh = d_3d_tmp7;
	float *d_boygr = d_3d_tmp8;
	float *d_stf = d_3d_tmp9;
	float *d_prod = d_3d_tmp10;
	float *d_dtef = d_3d_tmp11;

	float *d_dh = d_2d_tmp0;
	float *d_l0 = d_2d_tmp1;
	float *d_utau2 = d_2d_tmp2;
	*/
	float *d_utau2 = d_2d_tmp0;
	float *d_utau2_east = d_2d_tmp0_east;
	float *d_utau2_west = d_2d_tmp0_west;
	float *d_utau2_south = d_2d_tmp0_south;
	float *d_utau2_north = d_2d_tmp0_north;

	/*
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wusurf, wusurf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvsurf, wvsurf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2b, q2b, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2lb, q2lb, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u, u, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_v, v, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2, q2, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kh, kh, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_t, t, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_s, s, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rho, rho, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/


	//profq_gpu_kernel_0<<<blockPerGrid, threadPerBlock>>>(
	//		d_utau2, d_uf, 
	//		d_wusurf, d_wvsurf, d_wubot, d_wvbot,
	//		kb, jm, im);

	/*
	checkCudaErrors(cudaMemcpy(ee, d_ee, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gg, d_gg, jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(l0, d_l0, jm*im*sizeof(float), cudaMemcpyDeviceToHost));

	exchange2d_mpi(ee[0], im, jm);
	exchange2d_mpi(gg[0], im, jm);
	exchange2d_mpi(l0, im, jm);


	checkCudaErrors(cudaMemcpy(d_ee, ee, jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_gg, gg, jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l0, l0, jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	*/
	//exchange2d_mpi_gpu(d_utau2, im, jm);
	//exchange2d_cuda_aware_mpi(d_utau2, im, jm);


	profq_overlap_bcond_ew_gpu_kernel_0<<<
							blockPerGrid_ew_32, 
							threadPerBlock_ew_32,
							0, stream[1]>>>(
			d_utau2, d_wusurf, d_wvsurf, 
			jm, im);

	profq_overlap_bcond_sn_gpu_kernel_0<<<
							blockPerGrid_sn_32, 
							threadPerBlock_sn_32,
							0, stream[2]>>>(
			d_utau2, d_wusurf, d_wvsurf, 
			jm, im);

	////////////////////////////////////////////////////////
	//we use blockPerGrid, not blockPerGrid_inner here,
	//because we have to calculate all uf besides inner utau2 in kernel_0
	//
	//above is pervious implmentation, uf[kb-1] is moved to kernel_1
	////////////////////////////////////////////////////////

	profq_overlap_bcond_inner_gpu_kernel_0<<<
							   blockPerGrid_inner, 
							   threadPerBlock_inner, 
							   0, stream[0]>>>(
			d_utau2, d_wusurf, d_wvsurf, 
			kb, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

	profq_overlap_bcond_ew_bcond_gpu_kernel_0<<<
								  blockPerGrid_ew_b1, 
								  threadPerBlock_ew_b1,
								  0, stream[1]>>>(
			d_utau2, d_wusurf, d_wvsurf, 
			n_west, jm, im);

	profq_overlap_bcond_sn_bcond_gpu_kernel_0<<<
								  blockPerGrid_sn_b1, 
								  threadPerBlock_sn_b1,
							      0, stream[2]>>>(
			d_utau2, d_wusurf, d_wvsurf, 
			n_south, jm, im);


	//exchange2d_mpi_gpu(d_utau2, im, jm);

	exchange2d_cudaUVA(d_utau2, d_utau2_east, d_utau2_west,
					   d_utau2_south, d_utau2_north,
					   stream[1], im, jm);

	checkCudaErrors(cudaStreamSynchronize(stream[2]));

	//MPI_Barrier(pom_comm);
	//exchange2d_cuda_ipc(d_utau2, d_utau2_east, d_utau2_west,
	//				    stream[1], im, jm);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//checkCudaErrors(cudaStreamSynchronize(stream[2]));
	//MPI_Barrier(pom_comm);

	checkCudaErrors(cudaStreamSynchronize(stream[0]));

	//profq_gpu_kernel_1<<<blockPerGrid, threadPerBlock>>>(
	//		d_t, d_s, d_u, d_v, d_uf, d_vf, d_q2b, d_q2lb, d_q2,
	//		d_rho, d_etf, d_utau2,
	//		d_kq, d_km, d_kh, 
	//		d_h, d_z, d_zz, d_dz, d_dzz,
	//		grav, rhoref, kappa, tbias, sbias, dti2, small, umol, 
	//		kb, jm, im);

	profq_overlap_bcond_inner_gpu_kernel_1<<<
							   blockPerGrid_inner, 
							   threadPerBlock_inner,
							   0, stream[0]>>>(
			d_t, d_s, d_u, d_v, d_uf, d_vf, d_q2b, d_q2lb, d_q2,
			d_rho, d_etf, d_utau2,
			d_kq, d_km, d_kh, d_wubot, d_wvbot,
			d_fsm, d_h, d_z, d_zz, d_dz, d_dzz,
			grav, rhoref, kappa, tbias, sbias, dti2, small, umol, 
			kb, jm, im);

	profq_overlap_bcond_ew_gpu_kernel_1<<<
							blockPerGrid_ew_32, 
							threadPerBlock_ew_32,
							0, stream[1]>>>(
			d_t, d_s, d_u, d_v, d_uf, d_vf, d_q2b, d_q2lb, d_q2,
			d_rho, d_etf, d_utau2,
			d_kq, d_km, d_kh, d_wubot, d_wvbot,
			d_h, d_z, d_zz, d_dz, d_dzz,
			grav, rhoref, kappa, tbias, sbias, dti2, small, umol, 
			kb, jm, im);

	profq_overlap_bcond_sn_gpu_kernel_1<<<
							blockPerGrid_sn_32, 
							threadPerBlock_sn_32,
							0, stream[2]>>>(
			d_t, d_s, d_u, d_v, d_uf, d_vf, d_q2b, d_q2lb, d_q2,
			d_rho, d_etf, d_utau2,
			d_kq, d_km, d_kh, d_wubot, d_wvbot,
			d_h, d_z, d_zz, d_dz, d_dzz,
			grav, rhoref, kappa, tbias, sbias, dti2, small, umol, 
			kb, jm, im);

	profq_overlap_bcond_ew_bcond_gpu_kernel_1<<<
								  blockPerGrid_ew_b1, 
								  threadPerBlock_ew_b1,
								  0, stream[1]>>>(
			d_t, d_s, d_q2b, d_q2lb, d_q2,
			d_rho, d_etf, d_utau2,
			d_kq, d_km, d_kh, 
			d_h, d_z, d_zz, d_dz, d_dzz,
			grav, rhoref, kappa, tbias, sbias, dti2, small, umol, 
			kb, jm, im);

	profq_overlap_bcond_sn_bcond_gpu_kernel_1<<<
								  blockPerGrid_sn_b1, 
								  threadPerBlock_sn_b1,
								  0, stream[2]>>>(
			d_t, d_s, d_q2b, d_q2lb, d_q2,
			d_rho, d_etf, d_utau2,
			d_kq, d_km, d_kh, 
			d_h, d_z, d_zz, d_dz, d_dzz,
			grav, rhoref, kappa, tbias, sbias, dti2, small, umol, 
			kb, jm, im);



	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));
	//checkCudaErrors(cudaStreamSynchronize(stream[0]));

	/*
	checkCudaErrors(cudaMemcpy(km, d_km, kb*jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(kh, d_kh, kb*jm*im*sizeof(float), 
							   cudaMemcpyDeviceToHost));

    exchange3d_mpi(km,im,jm,kb);
    exchange3d_mpi(kh,im,jm,kb);

	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kh, kh, kb*jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	*/

    //exchange3d_mpi_gpu(d_km,im,jm,kb);
    //exchange3d_mpi_gpu(d_kh,im,jm,kb);

    //exchange3d_cuda_aware_mpi(d_km,im,jm,kb);
    //exchange3d_cuda_aware_mpi(d_kh,im,jm,kb);
	

	//profq_gpu_kernel_2<<<blockPerGrid, threadPerBlock>>>(
	//		d_km, d_kh, d_fsm,
	//		n_north, n_south, n_east, n_west, kb, jm, im);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));
	//checkCudaErrors(cudaStreamSynchronize(stream[2]));

	profq_overlap_bcond_ew_bcond_gpu_kernel_2<<<blockPerGrid_ew_b1, 
								  threadPerBlock_ew_b1,
								  0, stream[1]>>>(
			d_km, d_kh, d_fsm,
			n_east, n_west, kb, jm, im);

	//checkCudaErrors(cudaStreamSynchronize(stream[1]));

	profq_overlap_bcond_sn_bcond_gpu_kernel_2<<<blockPerGrid_sn_b1, 
								  threadPerBlock_sn_b1,
								  0, stream[2]>>>(
			d_km, d_kh, d_fsm,
			n_north, n_south, kb, jm, im);

	checkCudaErrors(cudaStreamSynchronize(stream[1]));
	checkCudaErrors(cudaStreamSynchronize(stream[2]));

	
	/*
	checkCudaErrors(cudaMemcpy(uf, d_uf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vf, d_vf, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(km, d_km, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(kh, d_kh, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(q2b, d_q2b, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(q2lb, d_q2lb, kb*jm*im*sizeof(float), cudaMemcpyDeviceToHost));
	*/

#ifndef TIME_DISABLE
		//checkCudaErrors(cudaDeviceSynchronize());
		timer_now(&end_profq);
		profq_time += time_consumed(&start_profq, 
									&end_profq);
#endif

	return;
}

