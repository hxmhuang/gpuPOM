#include"timer_all.h"

long long 
	advance_time				,
	surface_forcing_time		,
    momentum3d_time			    ,
    mode_interaction_time		,
    mode_external_time			,
	mode_internal_time			,
	print_section_time			,
	check_velocity_time			,
	store_mean_time             ,
	store_surf_mean_time        ,
	write_output_pnetcdf_time	,
	write_restart_pnetcdf_time	,
	advave_time					,
	advct_time					,
	advq_time					,
	advq_fusion_time			,
	advt1_time					,
	advt2_time					,
	advu_time					, 
	advv_time					,
	advuv_fusion_time			,
	baropg_time					, 
	baropg_mcc_time				,
	dens_time					,
	profq_time					,
	proft_time					,
	proft_fusion_time			,
	profu_time					,
	profv_time					,  
	profuv_fusion_time			,  
	vort_time					,  
	smol_adif_time				,
	vertvl_time					,
	realvertvl_time				,
	bcond_time   				,
	bcond_time_1   				,
	bcond_time_2   				,
	bcond_time_3   				,
	bcond_time_4   				,
	bcond_time_5   				,
	bcond_time_6   				,
	output_time   				,
	total_time   				,
	exchange3d_cuda_aware_mpi_time			,
	exchange3d_cuda_ipc_time			,
	exchange3d_mpi_time			,
	exchange2d_cuda_aware_mpi_time			,
	exchange2d_cuda_ipc_time			,
	exchange2d_mpi_time			,
	xperi2d_cuda_aware_mpi_time			,
	xperi2d_cuda_ipc_time			,
	xperi2d_mpi_time			,
	yperi2d_mpi_time			,
	xperi3d_cuda_aware_mpi_time			,
	xperi3d_cuda_ipc_time			,
	xperi3d_mpi_time			,
	yperi3d_mpi_time			;

inline void 
timer_now(struct timeval *time_now){

	gettimeofday(time_now, NULL);

}

inline long long  
time_consumed(struct timeval *time_begin,
		    			 struct timeval *time_end){
	return ((*time_end).tv_sec-(*time_begin).tv_sec)*1E+6
			+((*time_end).tv_usec-(*time_begin).tv_usec);
}
