#include<unistd.h>
#include<sys/types.h>
#include<pthread.h>

#include"data.h"
#include"cu_data.h"
#include"cu_data_io.h"
#include"cinitialize.h"
#include"cadvance.h"
#include"cadvance_gpu.h"
#include"cparallel_mpi.h"
#include"timer_all.h"
#include"cio_pnetcdf.h"

#include"cwind.h"
#include"criver.h"
#include"cassim.h"
#include"ctsforce.h"
#include"cmcsst.h"
#include"cassim_drf.h"
#include"ctrajdrf.h"
#include"cinterp.h"

int main(){

	//init_device();
	init_array();

	initialize();

	init_device_impi();

	if (origin_task < n_proc){//distinuish gpu_proc from io_proc
		if (calc_tsforce)
			tsforce_init();

		if (calc_tsurf_mc)
			mcsst_init();

		if (calc_interp)
			interp_init();
		
		wind_init();

		river_init();

		assim_init();

		assign_aid_array();

		init_cuda_1d_const();
		init_cuda_2d_const();
		init_cuda_3d_const();

		init_cuda_2d_var();
		init_cuda_3d_var();

		init_cuda_local();
		//init_cuda_ipc();
		exchangeMemHandle();
		openMemHandle();

		init_cuda_peer();
		init_cuda_pinned_memory();
	}
	init_cuda_ipc_io();

	//iend = 20;
	//iprint = 20;

	struct timeval time_start_output,
				   time_end_output;
	struct timeval time_start_total,
				   time_end_total;

	timer_now(&time_start_total);

	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	for (iint = 1; iint <= iend; iint++){

		if (origin_task < n_proc){//distinuish gpu_proc from io_proc
			printf("iint: %d\n", iint);
			if (calc_tsforce)
				tsforce_main();

			if (calc_tsurf_mc)
				mcsst_main();

			if (calc_tsforce)
				tsforce_tsflx();

			if (calc_wind)
				wind_main();

			if (calc_river)
				river_main();

			//advance();
			advance_gpu();

			if (calc_assimdrf)
				assimdrf_main();

			if (calc_assim)
				assim_main();

			if (calc_trajdrf)
				trajdrf_main();


			//if (iint % iprint == 0 && iint != iend){
			//	timer_now(&time_start_output);

			//	output_copy_back(num_out);
			//	write_output(num_out++);

			//	timer_now(&time_end_output);
			//	output_time += time_consumed(&time_start_output,
			//							 &time_end_output);
			//}
		}

		if (iint % iprint == 0 && iint != iend){
			timer_now(&time_start_output);
			MPI_Barrier(MPI_COMM_WORLD);
			if (origin_task >= n_proc){
				data_copy2H_io(num_out);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (origin_task >= n_proc){
				write_output(num_out++);
			}
			timer_now(&time_end_output);
			output_time += time_consumed(&time_start_output,
										 &time_end_output);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	timer_now(&time_end_total);
	total_time += time_consumed(&time_start_total,
								 &time_end_total);

	if (origin_task == master_task){
		printf("***********************************************\n");
		printf("%-30s:  %7.3lf, origin_rank:%d\n", "output_time",
				output_time/1.e+6, origin_task);	
		printf("%-30s:  %7.3lf, origin_rank:%d\n", "total_time",
				total_time/1.e+6, origin_task);	
		printf("***********************************************\n");
	}

	if (origin_task < n_proc){//distinuish gpu_proc from io_proc
		finalize_cuda_ipc();
    	finalize_cuda_gpu();	
		end_device();
	}

	finalize_mpi();
	//end_device();
	
	return 0;
}
