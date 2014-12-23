#include<mpi.h>

#include"data.h"
#include"cu_data_io.h"
#include"cu_data.h"

float *d_uab_mean_io;
float *d_vab_mean_io;
float *d_elb_mean_io;
float *d_wusurf_mean_io;
float *d_wvsurf_mean_io;
float *d_wtsurf_mean_io;
float *d_wssurf_mean_io;

float *d_uab_io;
float *d_vab_io;
float *d_elb_io;
float *d_et_io;
float *d_wusurf_io;
float *d_wvsurf_io;
float *d_wtsurf_io;
float *d_wssurf_io;

/////////////////////////

float *d_u_mean_io;
float *d_v_mean_io;
float *d_w_mean_io;
float *d_t_mean_io;
float *d_s_mean_io;
float *d_rho_mean_io;
float *d_kh_mean_io;
float *d_km_mean_io;

float *d_u_io;
float *d_v_io;
float *d_w_io;
float *d_t_io;
float *d_s_io;
float *d_rho_io;
float *d_kh_io;
float *d_km_io;

cudaIpcMemHandle_t handle_uab_mean_io, handle_uab_io,
				   handle_vab_mean_io, handle_vab_io,
				   handle_elb_mean_io, handle_elb_io,
				   handle_wusurf_mean_io, handle_wusurf_io,
				   handle_wvsurf_mean_io, handle_wvsurf_io,
				   handle_wtsurf_mean_io, handle_wtsurf_io,
				   handle_wssurf_mean_io, handle_wssurf_io,
				   handle_u_mean_io, handle_u_io,
				   handle_v_mean_io, handle_v_io,
				   handle_w_mean_io, handle_w_io,
				   handle_t_mean_io, handle_t_io,
				   handle_s_mean_io, handle_s_io,
				   handle_rho_mean_io, handle_rho_io,
				   handle_kh_mean_io, handle_kh_io,
				   handle_km_mean_io, handle_km_io,
				   handle_et_io;

void init_cuda_ipc_io(){


	MPI_Status status;

	/////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	//for the use of compute-io split
	//transfer relevant memory handle to io-process
	//whose origin_rank id equals rank+origin_size/2

	if (origin_task < n_proc){//compute-process
		checkCudaErrors(cudaIpcGetMemHandle(&handle_uab_mean_io, (void*)d_uab_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_vab_mean_io, (void*)d_vab_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_elb_mean_io, (void*)d_elb_mean));

		checkCudaErrors(cudaIpcGetMemHandle(&handle_wusurf_mean_io, (void*)d_wusurf_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wvsurf_mean_io, (void*)d_wvsurf_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wtsurf_mean_io, (void*)d_wtsurf_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wssurf_mean_io, (void*)d_wssurf_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_u_mean_io, (void*)d_u_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_v_mean_io, (void*)d_v_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_w_mean_io, (void*)d_w_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_t_mean_io, (void*)d_t_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_s_mean_io, (void*)d_s_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_rho_mean_io, (void*)d_rho_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_kh_mean_io, (void*)d_kh_mean));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_km_mean_io, (void*)d_km_mean));

		///////////////////////////////////////////////////////////////////

		checkCudaErrors(cudaIpcGetMemHandle(&handle_uab_io, (void*)d_uab));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_vab_io, (void*)d_vab));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_elb_io, (void*)d_elb));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_et_io, (void*)d_et));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wusurf_io, (void*)d_wusurf));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wvsurf_io, (void*)d_wvsurf));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wtsurf_io, (void*)d_wtsurf));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_wssurf_io, (void*)d_wssurf));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_u_io, (void*)d_u));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_v_io, (void*)d_v));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_w_io, (void*)d_w));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_t_io, (void*)d_t));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_s_io, (void*)d_s));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_rho_io, (void*)d_rho));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_kh_io, (void*)d_kh));
		checkCudaErrors(cudaIpcGetMemHandle(&handle_km_io, (void*)d_km));

		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////

		MPI_Send(&handle_uab_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_vab_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_elb_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wusurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wvsurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wtsurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wssurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_u_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_v_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_w_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_t_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_s_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_rho_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_kh_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_km_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);

		//////////////////////////////////////////////////////////////////

		MPI_Send(&handle_uab_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_vab_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_elb_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_et_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wusurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wvsurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wtsurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_wssurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_u_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_v_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_w_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_t_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_s_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_rho_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_kh_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&handle_km_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 origin_task+n_proc, 0, MPI_COMM_WORLD);
	}else{
		MPI_Recv(&handle_uab_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_vab_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_elb_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wusurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wvsurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wtsurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wssurf_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_u_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_v_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_w_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_t_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_s_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_rho_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_kh_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_km_mean_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	

		//////////////////////////////////////////////////////////////////

		MPI_Recv(&handle_uab_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_vab_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_elb_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_et_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wusurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wvsurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wtsurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_wssurf_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_u_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_v_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_w_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_t_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_s_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_rho_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_kh_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	
		MPI_Recv(&handle_km_io, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 origin_task-n_proc, 0, MPI_COMM_WORLD, &status);	

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uab_mean_io, handle_uab_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vab_mean_io, handle_vab_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_elb_mean_io, handle_elb_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wusurf_mean_io, handle_wusurf_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wvsurf_mean_io, handle_wvsurf_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wtsurf_mean_io, handle_wtsurf_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wssurf_mean_io, handle_wssurf_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_u_mean_io, handle_u_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_v_mean_io, handle_v_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_w_mean_io, handle_w_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_t_mean_io, handle_t_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_s_mean_io, handle_s_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_rho_mean_io, handle_rho_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_kh_mean_io, handle_kh_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_km_mean_io, handle_km_mean_io,
							 cudaIpcMemLazyEnablePeerAccess));

		//////////////////////////////////////////////////////////////////

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uab_io, handle_uab_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vab_io, handle_vab_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_elb_io, handle_elb_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_et_io, handle_et_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wusurf_io, handle_wusurf_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wvsurf_io, handle_wvsurf_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wtsurf_io, handle_wtsurf_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wssurf_io, handle_wssurf_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_u_io, handle_u_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_v_io, handle_v_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_w_io, handle_w_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_t_io, handle_t_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_s_io, handle_s_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_rho_io, handle_rho_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_kh_io, handle_kh_io,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_km_io, handle_km_io,
							 cudaIpcMemLazyEnablePeerAccess));
	}
}

void data_copy2H_io(int num_out){

	if (num_out == 0){
		checkCudaErrors(cudaMemcpy(uab, d_uab_io, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vab, d_vab_io, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(elb, d_elb_io, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(et, d_et_io, jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(u, d_u_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(v, d_v_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(w, d_w_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(t, d_t_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(s, d_s_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(rho, d_rho_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(kh, d_kh_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(km, d_km_io, kb*jm*im*sizeof(float), 
					cudaMemcpyDeviceToHost));
	}else{
		checkCudaErrors(cudaMemcpy(uab_mean, d_uab_mean_io, 
								   jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vab_mean, d_vab_mean_io, 
								   jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(elb_mean, d_elb_mean_io, 
								   jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(u_mean, d_u_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(v_mean, d_v_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(w_mean, d_w_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(t_mean, d_t_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(s_mean, d_s_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(rho_mean, d_rho_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(kh_mean, d_kh_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(km_mean, d_km_mean_io, 
								   kb*jm*im*sizeof(float), 
								   cudaMemcpyDeviceToHost));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}
