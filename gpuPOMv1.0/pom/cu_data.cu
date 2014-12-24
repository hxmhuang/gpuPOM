#include"cu_data.h"
#include"data.h"
#include"csolver_gpu.h"
#include<stdio.h>
#include<unistd.h>

//1D const 
float *d_zz;//added by momentum3d
float *d_dz;//added by mode_interaction 
float *d_uabe, *d_uabw, //added by mode_external
 	  *d_ele, *d_elw;
float *d_vabs, *d_vabn, 
	  *d_els, *d_eln;
float *d_dzz, *d_z;//added by mode_internal-->profq

//2D const
float *d_dx, *d_dy;//added by momentum3d 
float *d_aru, *d_arv;//added by momentum3d-->advct
float *d_aamfrz;//added by momentum3d 
float *d_art, *d_cor;//added by mode_external 
float *d_h, *d_fsm;//added by mode_external 
float *d_frz;//added by mode_internal-->bcond(4)


//3D const
float *d_tclim, *d_sclim;//added by mode_internal-->advt1


float *d_relax_aid;//added by mode_internal-->advt1

//2D
float *d_vfluxf, *d_e_atmos;//added by surface_forcing
float *d_swrad;//added by surface_forcing
float *d_dum, *d_dvm;//added by momentum3d 
float *d_dt, *d_d;//added by momentum3d 
float *d_adx2d, *d_ady2d;//added by mode_interaction 
float *d_drx2d, *d_dry2d;//added by mode_interaction 
float *d_aam2d;//added by mode_interaction 
float *d_ua, *d_va;//added by mode_interaction 
float *d_uab, *d_vab;//added by mode_interaction 
float *d_wubot, *d_wvbot;//added by mode_interaction 
float *d_advua, *d_advva;//added by mode_interaction 
float *d_el, *d_egf;//added by mode_interaction 
float *d_utf, *d_vtf;//added by mode_interaction 
float *d_cbc;//added by mode_interaction 

float *d_elb, *d_elf;//added by mode_external
float *d_wusurf, *d_wvsurf;//added by mode_external
float *d_etf;//added by mode_external
float *d_uaf, *d_vaf;//added by mode_external

float *d_utb, *d_vtb;//added by mode_internal
float *d_etb, *d_vfluxb;//added by mode_internal
float *d_tsurf;//added by mode_internal-->advt1
float *d_wtsurf, *d_wssurf;//added by mode_internal-->proft 
float *d_ssurf;//added by mode_internal-->proft

float *d_tbe, *d_sbe, *d_tbw, *d_sbw;//added by mode_internal-->bcond(4)
float *d_tbs, *d_sbs, *d_tbn, *d_sbn;//added by mode_internal-->bcond(4)

float *d_egb;//added by mode_internal-->advu
float *d_et;//added by mode_internal-->kernel_4

float *d_uab_mean, *d_vab_mean;//added by store_mean
float *d_elb_mean;//added by store_mean
float *d_wusurf_mean, *d_wvsurf_mean;//added by store_mean
float *d_wtsurf_mean, *d_wssurf_mean;//added by store_mean

float *d_usrf_mean, *d_vsrf_mean;//add by store_surf_mean
float *d_elsrf_mean;//add by store_surf_mean 
float *d_uwsrf_mean, *d_vwsrf_mean;//add by store_surf_mean
float *d_uwsrf, *d_vwsrf;//add by store_surf_mean
float *d_utf_mean, *d_vtf_mean;//add by store_surf_mean
float *d_celg_mean, *d_ctsurf_mean;//add by store_surf_mean
float *d_celg, *d_ctsurf;//add by store_surf_mean
float *d_cpvf_mean, *d_cjbar_mean;//add by store_surf_mean
float *d_cpvf, *d_cjbar;//add by store_surf_mean
float *d_cadv_mean, *d_cten_mean;//add by store_surf_mean
float *d_cadv, *d_cten;//add by store_surf_mean
float *d_ctbot_mean;//add by store_surf_mean
float *d_ctbot;//add by store_surf_mean

float *d_ctot, *d_totx, *d_toty;//add by mode_external-->vort

//3D
float *d_w;//added by surface_forcing
float *d_u, *d_v;//added by momentum3d 
float *d_ub, *d_vb;//added by momentum3d 
float *d_aam, *d_rho, *d_rmean;//added by momentum3d 
float *d_advx, *d_advy;//added by momentum3d 
float *d_drhox, *d_drhoy;//added by momentum3d 

float *d_q2b, *d_q2;//added by mode_internal 
float *d_q2lb, *d_q2l;//added by mode_internal 
float *d_uf, *d_vf;//added by mode_internal 
float *d_kq, *d_l;//added by mode_internal-->profq 
float *d_t, *d_s;//added by mode_internal-->profq 
float *d_km, *d_kh;//added by mode_internal-->profq 
float *d_tb, *d_sb; //added by mode_internal-->advt1

float *d_tobw, *d_sobw, //added by mode_internal-->bcond(4)
	  *d_tobe, *d_sobe,
	  *d_tobs, *d_sobs,
	  *d_tobn, *d_sobn;

float *d_u_mean, *d_v_mean, *d_w_mean;//added by store_mean
float *d_t_mean, *d_s_mean;//added by store_mean
float *d_rho_mean, *d_kh_mean, *d_km_mean;//added by store_mean
float *d_ustks, *d_vstks;//added by store_mean

float *d_xstks_mean, *d_ystks_mean;//add by store_surf_mean
float *d_xstks, *d_ystks;//add by store_surf_mean

float *d_3d_tmp0, *d_3d_tmp1, 
	  *d_3d_tmp2, *d_3d_tmp3;

float *d_3d_tmp4, *d_3d_tmp5, 
	  *d_3d_tmp6, *d_3d_tmp7,
	  *d_3d_tmp8, *d_3d_tmp9,
	  *d_3d_tmp10, *d_3d_tmp11;

float *d_2d_tmp0, *d_2d_tmp1;

float *d_2d_tmp2, *d_2d_tmp3,
	  *d_2d_tmp4, *d_2d_tmp5,
	  *d_2d_tmp6, *d_2d_tmp7,
	  *d_2d_tmp8, *d_2d_tmp9,
	  *d_2d_tmp10, *d_2d_tmp11,
	  *d_2d_tmp12, *d_2d_tmp13;

float *d_1d_ny_tmp0, *d_1d_ny_tmp1,
	  *d_1d_ny_tmp2, *d_1d_ny_tmp3;

//////////////////////////////////////////////
//data copy back to host and then mpi
float *h_1d_nx_tmp0, *h_1d_nx_tmp1,
	  *h_1d_nx_tmp2, *h_1d_nx_tmp3;
float *h_1d_ny_tmp0, *h_1d_ny_tmp1,
	  *h_1d_ny_tmp2, *h_1d_ny_tmp3;
//////////////////////////////////////////////

float *d_2d_ny_nz_tmp0, *d_2d_ny_nz_tmp1,
 	  *d_2d_ny_nz_tmp2, *d_2d_ny_nz_tmp3;

float *d_2d_nx_nz_tmp0, *d_2d_nx_nz_tmp1,
	  *d_2d_nx_nz_tmp2, *d_2d_nx_nz_tmp3;


//////////////////////////////////////////////
//data copy back to host and then mpi
float *h_2d_ny_nz_tmp0, *h_2d_ny_nz_tmp1,
 	  *h_2d_ny_nz_tmp2, *h_2d_ny_nz_tmp3;

float *h_2d_nx_nz_tmp0, *h_2d_nx_nz_tmp1,
 	  *h_2d_nx_nz_tmp2, *h_2d_nx_nz_tmp3;
//////////////////////////////////////////////


///////////////////////////////////////
//for the use of cudaIpc
float *d_ctsurf_east, *d_ctsurf_west,
 	  *d_ctbot_east, *d_ctbot_west,
 	  *d_celg_east, *d_celg_west,
 	  *d_cjbar_east, *d_cjbar_west,
 	  *d_cadv_east, *d_cadv_west,
 	  *d_cpvf_east, *d_cpvf_west,
 	  *d_cten_east, *d_cten_west;

/////////////////////////////////////////////
float *d_2d_tmp0_east, *d_2d_tmp0_west,
	  *d_2d_tmp1_east, *d_2d_tmp1_west,
	  *d_2d_tmp2_east, *d_2d_tmp2_west,
	  *d_2d_tmp3_east, *d_2d_tmp3_west,
	  *d_2d_tmp4_east, *d_2d_tmp4_west,
	  *d_2d_tmp5_east, *d_2d_tmp5_west,
	  *d_2d_tmp6_east, *d_2d_tmp6_west,
	  *d_2d_tmp7_east, *d_2d_tmp7_west,
	  *d_2d_tmp8_east, *d_2d_tmp8_west,
	  *d_2d_tmp9_east, *d_2d_tmp9_west,
	  *d_2d_tmp10_east, *d_2d_tmp10_west,
	  *d_2d_tmp11_east, *d_2d_tmp11_west,
	  *d_2d_tmp12_east, *d_2d_tmp12_west,
	  *d_2d_tmp13_east, *d_2d_tmp13_west;

float *d_2d_tmp0_south, *d_2d_tmp0_north,
	  *d_2d_tmp1_south, *d_2d_tmp1_north;
/////////////////////////////////////////////

float *d_totx_east, *d_totx_west,
	  *d_toty_east, *d_toty_west;

/////////////////////////////////////////////
float *d_3d_tmp0_east, *d_3d_tmp0_west,
	  *d_3d_tmp1_east, *d_3d_tmp1_west,
	  *d_3d_tmp2_east, *d_3d_tmp2_west;

float *d_3d_tmp0_south, *d_3d_tmp0_north,
	  *d_3d_tmp1_south, *d_3d_tmp1_north,
	  *d_3d_tmp2_south, *d_3d_tmp2_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_wubot_east, *d_wubot_west,
	  *d_wubot_east_most, *d_wubot_west_most,
	  *d_wvbot_east, *d_wvbot_west,
	  *d_wvbot_east_most, *d_wvbot_west_most;

float *d_wubot_south, *d_wubot_north,
	  *d_wvbot_south, *d_wvbot_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_aam_east, *d_aam_west;
float *d_aam_south, *d_aam_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_elf_east, *d_elf_west;
float *d_elf_east_most, *d_elf_west_most;
float *d_elf_south, *d_elf_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_uaf_east, *d_uaf_west;
float *d_uaf_east_most, *d_uaf_west_most;
float *d_uaf_south, *d_uaf_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_vaf_east, *d_vaf_west;
float *d_vaf_east_most, *d_vaf_west_most;
float *d_vaf_south, *d_vaf_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_w_east, *d_w_west;
float *d_w_east_most, *d_w_west_most;
float *d_w_south, *d_w_north;
/////////////////////////////////////////////


/////////////////////////////////////////////
float *d_uf_east, *d_uf_west,
	  *d_uf_south, *d_uf_north,
	  *d_vf_east, *d_vf_west,
	  *d_vf_south, *d_vf_north,
	  *d_uf_east_most, *d_uf_west_most,
	  *d_vf_east_most, *d_vf_west_most,
	  *d_kh_east_most, *d_kh_west_most,
	  *d_km_east_most, *d_km_west_most,
	  *d_kq_east_most, *d_kq_west_most;
/////////////////////////////////////////////

cudaIpcMemHandle_t handle_ctsurf,
				   handle_ctbot,
				   handle_celg,
				   handle_cjbar,
				   handle_cadv,
				   handle_cpvf,
				   handle_cten,
				   handle_ctsurf_east, handle_ctsurf_west,
			       handle_ctbot_east, handle_ctbot_west,
			       handle_celg_east, handle_celg_west,
			       handle_cjbar_east, handle_cjbar_west,
			       handle_cadv_east, handle_cadv_west,
			       handle_cpvf_east, handle_cpvf_west,
			       handle_cten_east, handle_cten_west;

/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_2d_tmp0, handle_2d_tmp1,
				   handle_2d_tmp2, handle_2d_tmp3,
				   handle_2d_tmp4, handle_2d_tmp5,
				   handle_2d_tmp6, handle_2d_tmp7,
				   handle_2d_tmp8, handle_2d_tmp9,
				   handle_2d_tmp10, handle_2d_tmp11,
				   handle_2d_tmp12, handle_2d_tmp13,
				   handle_2d_tmp0_east, handle_2d_tmp0_west,
				   handle_2d_tmp0_south, handle_2d_tmp0_north,
				   handle_2d_tmp1_east, handle_2d_tmp1_west,
				   handle_2d_tmp1_south, handle_2d_tmp1_north,
				   handle_2d_tmp2_east, handle_2d_tmp2_west,
				   handle_2d_tmp3_east, handle_2d_tmp3_west,
				   handle_2d_tmp4_east, handle_2d_tmp4_west,
				   handle_2d_tmp5_east, handle_2d_tmp5_west,
				   handle_2d_tmp6_east, handle_2d_tmp6_west,
				   handle_2d_tmp7_east, handle_2d_tmp7_west,
				   handle_2d_tmp8_east, handle_2d_tmp8_west,
				   handle_2d_tmp9_east, handle_2d_tmp9_west,
				   handle_2d_tmp10_east, handle_2d_tmp10_west,
				   handle_2d_tmp11_east, handle_2d_tmp11_west,
				   handle_2d_tmp12_east, handle_2d_tmp12_west,
				   handle_2d_tmp13_east, handle_2d_tmp13_west;
/////////////////////////////////////////////////////////////////

cudaIpcMemHandle_t handle_totx, handle_toty,
				   handle_totx_east, handle_totx_west,
				   handle_toty_east, handle_toty_west;

/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_3d_tmp0, handle_3d_tmp1, handle_3d_tmp2,
				   handle_3d_tmp0_east, handle_3d_tmp0_west,
				   handle_3d_tmp0_south, handle_3d_tmp0_north,
				   handle_3d_tmp1_east, handle_3d_tmp1_west,
				   handle_3d_tmp1_south, handle_3d_tmp1_north,
				   handle_3d_tmp2_east, handle_3d_tmp2_west,
				   handle_3d_tmp2_south, handle_3d_tmp2_north;
/////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_wubot, handle_wvbot,
				   handle_wubot_east, handle_wubot_west,
				   handle_wubot_south, handle_wubot_north,
				   handle_wubot_east_most, handle_wubot_west_most,
				   handle_wvbot_east, handle_wvbot_west,
				   handle_wvbot_south, handle_wvbot_north,
				   handle_wvbot_east_most, handle_wvbot_west_most;
/////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_aam,
				   handle_aam_east, handle_aam_west,
				   handle_aam_south, handle_aam_north;
/////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_elf,
				   handle_elf_east, handle_elf_west,
				   handle_elf_south, handle_elf_north,
				   handle_elf_east_most, handle_elf_west_most;
/////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_uaf, handle_vaf,
				   handle_uaf_east, handle_uaf_west,
				   handle_uaf_south, handle_uaf_north,
				   handle_vaf_east, handle_vaf_west,
				   handle_vaf_south, handle_vaf_north,
				   handle_uaf_east_most, handle_uaf_west_most,
				   handle_vaf_east_most, handle_vaf_west_most;
/////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_w,
				   handle_w_east, handle_w_west,
				   handle_w_south, handle_w_north,
				   handle_w_east_most, handle_w_west_most;
/////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////
cudaIpcMemHandle_t handle_uf, handle_vf,
				   handle_kh, handle_km, handle_kq,
				   handle_uf_east, handle_uf_west,
				   handle_uf_south, handle_uf_north,
				   handle_vf_east, handle_vf_west,
				   handle_vf_south, handle_vf_north,
				   handle_uf_east_most, handle_uf_west_most,
				   handle_vf_east_most, handle_vf_west_most,
				   handle_kh_east_most, handle_kh_west_most,
				   handle_km_east_most, handle_km_west_most,
				   handle_kq_east_most, handle_kq_west_most;
/////////////////////////////////////////////////////////////////



cudaStream_t stream[5];

dim3 threadPerBlock(block_i_2D, block_j_2D);
dim3 blockPerGrid((i_size+block_i_2D-1)/block_i_2D,
				  (j_size+block_j_2D-1)/block_j_2D);

dim3 threadPerBlock_inner(block_i_2D, block_j_2D);
dim3 blockPerGrid_inner((i_size-2-64+block_i_2D-1)/block_i_2D,
					    (j_size-2-64+block_j_2D-1)/block_j_2D);

dim3 threadPerBlock_ew_32(32, 4);
dim3 blockPerGrid_ew_32(2, (j_size-2+3)/4);

dim3 threadPerBlock_sn_32(32, 4);
dim3 blockPerGrid_sn_32((i_size-2+31)/32, 16);

dim3 threadPerBlock_ew_b1(1, 128);
dim3 blockPerGrid_ew_b1(1, (j_size+127)/128);

dim3 threadPerBlock_ew_b2(1, 128);
dim3 blockPerGrid_ew_b2(2, (j_size+127)/128);

dim3 threadPerBlock_sn_b1(128, 1);
dim3 blockPerGrid_sn_b1((i_size+127)/128, 1);

dim3 threadPerBlock_sn_b2(128, 1);
dim3 blockPerGrid_sn_b2((i_size+127)/128, 2);


/*
//FILE *out;

//persistent variables
//1D const

float *d_dzz, *d_dz, *d_zz, *d_z;
int *d_j_global;
float *d_uabe, *d_uabw, *d_ele, *d_elw;
float *d_vabs, *d_vabn, *d_els, *d_eln;

float *d_aam_aid;

//2D const
float *d_fsm, *d_aru, *d_arv, *d_art;
float *d_dx, *d_dy, *d_dum, *d_dvm;
float *d_cor, *d_cbc,  *d_h;
float *d_tsurf, *d_ssurf;
float *d_tbe, *d_sbe, *d_tbw, *d_sbw;
float *d_tbs, *d_sbs, *d_tbn, *d_sbn;

//3D const
float *d_rmean, *d_tclim, *d_sclim;
/////////////////////////////////////////////////////

float *d_dt;
float *d_etf, *d_aam;



float *d_u, *d_v, *d_etb, *d_w;//added by advt2
float *d_advua, *d_advva, *d_d, *d_ua, *d_va, 
 	  *d_fluxua, *d_fluxva, *d_uab, *d_vab, *d_aam2d;//added by advave
float *d_advx, *d_advy, *d_ub, *d_vb; //added by advct
float *d_egf, *d_egb, *d_e_atmos, *d_drhox, *d_uf;//added by advu
float *d_drhoy, *d_vf;//added by advv
float *d_rho;//added by baropg 
float *d_vfluxf, *d_vfluxb;//added by vertvl
float *d_kq, *d_wusurf, *d_wvsurf, *d_wubot, *d_wvbot;//added by profq
float *d_t, *d_s, *d_q2b, *d_q2lb, *d_l, *d_km, *d_kh, *d_q2;//added by profq
float *d_swrad;//added by proft
float *d_wtsurf, *d_wssurf;//added by surface_forcing 
float *d_el;//added by mode_interaction
float *d_elb;//added by mode_external
float *d_utb, *d_vtb, *d_q2l;//added by mode_internal
float *d_tb, *d_sb, *d_et; //added by mode_internal

/////////////////////////////////////////////////////
//variable
//need not copy-in
float *d_adx2d, *d_ady2d, *d_drx2d, *d_dry2d; //added by mode_interaction;
float *d_elf, *d_utf, *d_vtf; //added by mode_external
float *d_uaf, *d_vaf; //added by mode_external
*/



/*
//local variables
float *d_3d_tmp0, *d_3d_tmp1, *d_3d_tmp2, 
	  *d_3d_tmp3, *d_3d_tmp4, *d_3d_tmp5,
	  *d_3d_tmp6; //*d_3d_tmp7, *d_3d_tmp8,
	  //*d_3d_tmp9, *d_3d_tmp10, *d_3d_tmp11,
	  //*d_3d_tmp12;


//double *d_3d_tmp0_d, *d_3d_tmp1_d, 
//	   *d_3d_tmp2_d, *d_3d_tmp3_d;


float *d_2d_tmp0, *d_2d_tmp1, *d_2d_tmp2; 
*/

void check(cudaError_t err, const char* file, const char* func, unsigned line){
	if (err != cudaSuccess){
		fprintf(stderr, "Hello: ERR: %s, file is %s, func is %s, line is %u, my_task is %d\n", 
				cudaGetErrorString(err), file, func, line, my_task);	
		exit(1);
	}
}

void init_cuda_scalar_const_(int n_east, int n_west, int n_north, int n_south,
							 int my_task, int kb, int jm, int im,
							 int nitera, int mode, int ntp,
							 float sw, float dit2, float tprni, float grav, 
							 float tbias, float sbias, float rhoref, float umol,
							 float kappa, float small){

	return;
}


void init_cuda_1d_const(){
	
	checkCudaErrors(cudaMalloc((void**) &d_zz,  kb*sizeof(float)));

	//added by mode_interaction 
	checkCudaErrors(cudaMalloc((void**) &d_dz,  kb*sizeof(float)));

	//added by mode_external
	checkCudaErrors(cudaMalloc((void**) &d_uabe, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uabw, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ele, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elw, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vabs, im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vabn, im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_els, im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_eln, im*sizeof(float)));

	//added by mode_internal--profq
	checkCudaErrors(cudaMalloc((void**) &d_dzz,  kb*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_z, kb*sizeof(float)));
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	checkCudaErrors(cudaMemcpy(d_zz,  zz,  kb*sizeof(float), 
							   cudaMemcpyHostToDevice));

	//added by mode_interaction 
	checkCudaErrors(cudaMemcpy(d_dz,  dz,  kb*sizeof(float), 
							   cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_uabe, uabe, jm*sizeof(float), 
						       cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uabw, uabw, jm*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ele, ele, jm*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_elw, elw, jm*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vabs, vabs, im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vabn, vabn, im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_els, els, im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_eln, eln, im*sizeof(float), 
							   cudaMemcpyHostToDevice));


	//added by mode_internal--profq
	checkCudaErrors(cudaMemcpy(d_z,   z,   kb*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dzz, dzz, kb*sizeof(float), 
							   cudaMemcpyHostToDevice));
	/*
	checkCudaErrors(cudaMalloc((void**) &d_z,   kb*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_zz,  kb*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dz,  kb*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dzz, kb*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**) &d_j_global, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uabe, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uabw, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ele, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elw, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vabs, im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vabn, im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_els, im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_eln, im*sizeof(float)));


	checkCudaErrors(cudaMemcpy(d_z,   z,   kb*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_zz,  zz,  kb*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dz,  dz,  kb*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dzz, dzz, kb*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_j_global, j_global, jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uabe, uabe, jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uabw, uabw, jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ele, ele, jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_elw, elw, jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vabs, vabs, im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vabn, vabn, im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_els, els, im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_eln, eln, im*sizeof(float), cudaMemcpyHostToDevice));

	initial_constant_csolver(dzz, dz, zz, z);
	*/
	return;

}

/*
void init_cuda_2d_const(float* aru, float* arv, float *art,
						 float *dx, float *dy,
						 float *dum, float *dvm, 
						 float* h, float *cor, 
						 float* fsm, float *cbc,
						 float *tsurf, float *ssurf,
						 float *tbe, float *sbe,
						 float *tbw, float *sbw,
						 float *tbs, float *sbs,
						 float *tbn, float *sbn){
*/
void init_cuda_2d_const(){

	//added by momentum3d
	checkCudaErrors(cudaMalloc((void**) &d_dx,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dy,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_aamfrz,  jm*im*sizeof(float)));
	//added by momentum3d-->advct
	checkCudaErrors(cudaMalloc((void**) &d_aru, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_arv, jm*im*sizeof(float)));
	//added by mode_external
	checkCudaErrors(cudaMalloc((void**) &d_art, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cor, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_h, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_fsm, jm*im*sizeof(float)));

	//added by mode_internal-->bcond(4)
	checkCudaErrors(cudaMalloc((void**) &d_frz, jm*im*sizeof(float)));




	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////


	checkCudaErrors(cudaMemcpy(d_dx,  dx,  jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dy,  dy,  jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aamfrz,  aamfrz,  jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	//added by momentum3d-->advct
	checkCudaErrors(cudaMemcpy(d_aru, aru, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arv, arv, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));
	//added by mode_external
	checkCudaErrors(cudaMemcpy(d_art, art, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cor, cor, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_h, h, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fsm, fsm, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));

	//added by mode_internal-->bcond(4)
	checkCudaErrors(cudaMemcpy(d_frz, frz, jm*im*sizeof(float), 
								cudaMemcpyHostToDevice));



	/*
	//below are for persistent global variables memroy alloc
	checkCudaErrors(cudaMalloc((void**) &d_aru, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_arv, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_art, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dx,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dy,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dum, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dvm, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_h,   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cor, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_fsm, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cbc, jm*im*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**) &d_tsurf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ssurf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbe, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbe, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbw, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbw, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbs, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbs, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbn, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbn, kb*im*sizeof(float)));
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	checkCudaErrors(cudaMemcpy(d_aru, aru, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arv, arv, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_art, art, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dx,  dx,  jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dy,  dy,  jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dum, dum, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dvm, dvm, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_h,   h,   jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cor, cor, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fsm, fsm, jm*im*sizeof(float), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMemcpy(d_cbc, cbc, jm*im*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_tsurf, tsurf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ssurf, ssurf, jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbe, tbe, kb*jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbe, sbe, kb*jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbw, tbw, kb*jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbw, sbw, kb*jm*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbs, tbs, kb*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbs, sbs, kb*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbn, tbn, kb*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbn, sbn, kb*im*sizeof(float), cudaMemcpyHostToDevice));
	*/

	return;
}


void init_cuda_3d_const(){

	//added by mode_internal-->advt1
	checkCudaErrors(cudaMalloc((void**) &d_tclim, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sclim, kb*jm*im*sizeof(float)));

	//added by mode_internal-->advt1
	checkCudaErrors(cudaMalloc((void**) &d_relax_aid, kb*jm*im*sizeof(float)));

	////////////////////////////////////////////////

	checkCudaErrors(cudaMemcpy(d_tclim, tclim, kb*jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sclim, sclim, kb*jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));


	//added by mode_internal-->advt1
	checkCudaErrors(cudaMemcpy(d_relax_aid, relax_aid, 
							   kb*jm*im*sizeof(float), 
							   cudaMemcpyHostToDevice));
	/*
	checkCudaErrors(cudaMalloc((void**) &d_rmean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tclim, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sclim, kb*jm*im*sizeof(float)));
	////////////////////////////////////////////////
	//kq is a special variable, because it is only referenced and modified in profq
	//but we have to make it a global variable for the value is useful in the next iteration
	////////////////////////////////////////////////
	checkCudaErrors(cudaMalloc((void**) &d_kq,    kb*jm*im*sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_rmean, rmean, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tclim, tclim, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sclim, sclim, kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kq,    kq,    kb*jm*im*sizeof(float), cudaMemcpyHostToDevice));
	*/

	return;
}


void init_cuda_1d_var(){

	/*
	checkCudaErrors(cudaMalloc((void**) &d_aam_aid, jm*sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_aam_aid, aam_aid, jm*sizeof(float), cudaMemcpyHostToDevice));
	*/
	return;
}

/*
void init_cuda_2d_var(float *dt, float *el,
					  float *ua, float *va,
					  float *d, float *uab, 
					  float *vab, float *aam2d,
					  float *wubot, float *wvbot, 
					  float *advua, float *advva,
					  float *adx2d, float *ady2d,
					  float *drx2d, float *dry2d,
					  float *elb, float *etf,
					  float *utb, float *vtb,
					  float *etb, float *vfluxb,
					  float *egb, float *et){
*/
void init_cuda_2d_var(){

	//added by surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_vfluxf, 
							   jm*im*sizeof(float)));
	//assigned 0 in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_e_atmos, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_swrad, 
							   jm*im*sizeof(float)));

	//added by momentum3d
	checkCudaErrors(cudaMalloc((void**) &d_dum, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dvm, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dt, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_d, 
							   jm*im*sizeof(float)));
	//added by mode_interaction 
	checkCudaErrors(cudaMalloc((void**) &d_adx2d, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ady2d, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_drx2d, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dry2d, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_aam2d, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ua, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_va, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uab, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vab, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wubot, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wvbot, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advua, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advva, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_el, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_egf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_utf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vtf, 
							   jm*im*sizeof(float)));
	//added by mode_interaction --advave
	checkCudaErrors(cudaMalloc((void**) &d_cbc, 
							   jm*im*sizeof(float)));

	//added by mode_external
	checkCudaErrors(cudaMalloc((void**) &d_elb, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wusurf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wvsurf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_etf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uaf, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vaf, 
							   jm*im*sizeof(float)));

	//added by mode_internal
	checkCudaErrors(cudaMalloc((void**) &d_utb, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vtb, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_etb, 
							   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vfluxb, 
							   jm*im*sizeof(float)));

	//added by mode_internal-->advt1
	checkCudaErrors(cudaMalloc((void**) &d_tsurf, jm*im*sizeof(float)));

	//added by mode_internal-->proft
	checkCudaErrors(cudaMalloc((void**) &d_wtsurf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wssurf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ssurf, jm*im*sizeof(float)));

	//added by mode_internal-->bcond(4)
	checkCudaErrors(cudaMalloc((void**) &d_tbe, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbe, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbw, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbw, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbs, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbs, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tbn, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sbn, kb*im*sizeof(float)));

	//added by mode_internal-->advu
	checkCudaErrors(cudaMalloc((void**) &d_egb, jm*im*sizeof(float)));

	//added by mode_internal-->kernel_4
	checkCudaErrors(cudaMalloc((void**) &d_et, jm*im*sizeof(float)));

	//added by store_mean
	checkCudaErrors(cudaMalloc((void**) &d_uab_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vab_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elb_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wusurf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wvsurf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wtsurf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wssurf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_usrf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vsrf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elsrf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uwsrf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vwsrf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_utf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vtf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uwsrf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vwsrf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_celg_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ctsurf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_celg, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ctsurf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cpvf_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cjbar_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cpvf, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cjbar, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cadv_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cten_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cadv, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_cten, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ctbot_mean, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ctbot, jm*im*sizeof(float)));

	//added by mode_external-->vort 
	checkCudaErrors(cudaMalloc((void**) &d_ctot, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_totx, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_toty, jm*im*sizeof(float)));

    /////////////////////////////////////////////////////

	checkCudaErrors(cudaMemcpy(d_vfluxf, vfluxf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dum, dum, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dvm, dvm, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d, d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
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
	checkCudaErrors(cudaMemcpy(d_ua, ua, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_va, va, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uab, uab, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vab, vab, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advua, advua, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advva, advva, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_el, el, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cbc, cbc, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_elb, elb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wusurf, wusurf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvsurf, wvsurf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));


	//added by mode_internal
	checkCudaErrors(cudaMemcpy(d_utb, utb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vtb, vtb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfluxb, vfluxb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by mode_internal-->proft
	checkCudaErrors(cudaMemcpy(d_wtsurf, wtsurf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wssurf, wssurf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by mode_internal-->bcond(4)
	checkCudaErrors(cudaMemcpy(d_tbe, tbe, kb*jm*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbe, sbe, kb*jm*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbw, tbw, kb*jm*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbw, sbw, kb*jm*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbs, tbs, kb*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbs, sbs, kb*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tbn, tbn, kb*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sbn, sbn, kb*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by mode_internal-->advu
	checkCudaErrors(cudaMemcpy(d_egb, egb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by mode_internal-->kernel_4
	checkCudaErrors(cudaMemcpy(d_et, et, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////

	checkCudaErrors(cudaMemset(d_e_atmos, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_swrad, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_egf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_utf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vtf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_elf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_uaf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vaf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_tsurf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ssurf, 0, jm*im*sizeof(float)));

	//added by store_mean
	checkCudaErrors(cudaMemset(d_uab_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vab_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_elb_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wusurf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wvsurf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wtsurf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wssurf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_usrf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vsrf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_elsrf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_uwsrf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vwsrf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_celg_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ctsurf_mean, 0, jm*im*sizeof(float)));

	checkCudaErrors(cudaMemset(d_celg, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ctsurf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cpvf_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cjbar_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cpvf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cjbar, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cadv_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cten_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cadv, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_cten, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ctbot_mean, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ctbot, 0, jm*im*sizeof(float)));

	//added by mode_external-->vort
	checkCudaErrors(cudaMemset(d_ctot, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_totx, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_toty, 0, jm*im*sizeof(float)));

	/*
	checkCudaErrors(cudaMalloc((void**) &d_dt,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_el,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ua,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_va,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_d,       jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uab,     jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vab,     jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_aam2d,   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wubot,   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_wvbot,   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advua,   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advva,   jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_adx2d,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ady2d,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_drx2d,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_dry2d,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elb,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_etf,     jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_utb,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vtb,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_etb,     jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vfluxb,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_egb,     jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_et,      jm*im*sizeof(float)));



    /////////////////////////////////////////////////////
	//variable
	//need not copy-in
	//egf need not copied in for it is assigned in mode_interaction
	checkCudaErrors(cudaMalloc((void**) &d_utf,      jm*im*sizeof(float)));
	//utf need not copied in for it is assigned in mode_interaction
	checkCudaErrors(cudaMalloc((void**) &d_egf,     jm*im*sizeof(float)));
	//vtf need not copied in for it is assigned in mode_interaction
	checkCudaErrors(cudaMalloc((void**) &d_vtf,      jm*im*sizeof(float)));
	//vfluxf need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_vfluxf,  jm*im*sizeof(float)));
	//e_atmos need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_e_atmos, jm*im*sizeof(float)));
	//wusurf need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_wusurf,  jm*im*sizeof(float)));
	//wvsurf need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_wvsurf,  jm*im*sizeof(float)));
	//wtsurf need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_wtsurf,      jm*im*sizeof(float)));
	//wssurf need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_wssurf,      jm*im*sizeof(float)));
	//swrad need not copied in for it is assigned in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_swrad,   jm*im*sizeof(float)));

	//checkCudaErrors(cudaMalloc((void**) &d_dry2d,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_fluxua,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_fluxva,  jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_elf,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uaf,      jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vaf,      jm*im*sizeof(float)));



	//added by surface forcing
	checkCudaErrors(cudaMemcpy(d_dt, dt, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	//added by mode_interaction
	checkCudaErrors(cudaMemcpy(d_el, el, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ua, ua, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_va, va, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d, d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uab, uab, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vab, vab, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aam2d, aam2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wubot, wubot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wvbot, wvbot, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advua, advua, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advva, advva, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	//comment: below are only assigned 0, but in iopnetcdf,
	//they are read, I am not sure whether below are useful,
	//now I just read them;
	checkCudaErrors(cudaMemcpy(d_adx2d, adx2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ady2d, ady2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drx2d, drx2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_dry2d, dry2d, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by mode_external
	checkCudaErrors(cudaMemcpy(d_elb, elb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etf, etf, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by mode_internal
	checkCudaErrors(cudaMemcpy(d_utb, utb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vtb, vtb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_etb, etb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vfluxb, vfluxb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_egb, egb, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_et, et, jm*im*sizeof(float), 
					cudaMemcpyHostToDevice));

	//added by surface-forcing
	checkCudaErrors(cudaMemset(d_wusurf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wvsurf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wtsurf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wssurf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_e_atmos, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_swrad, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vfluxf, 0, jm*im*sizeof(float)));

	checkCudaErrors(cudaMemset(d_utf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_egf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vtf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_fluxua, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_fluxva, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_elf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_uaf, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_vaf, 0, jm*im*sizeof(float)));
	*/

	return;
}


/*
void init_cuda_3d_var(float *t, float *s,
					  float *u, float *v,
					  float *ub, float *vb,
					  float *aam, float *rho,
					  float *advx, float *advy,
					  float *drhox, float *drhoy,
					  float *q2b, float *q2,
					  float *q2lb, float *q2l,
					  float *tb, float *sb, 
					  float *km, float *kh){
*/
void init_cuda_3d_var(){

	//re_assigned vfluxf in surface_forcing
	checkCudaErrors(cudaMalloc((void**) &d_w,     kb*jm*im*sizeof(float)));

	//added by momentum3d
	checkCudaErrors(cudaMalloc((void**) &d_u,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_v,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ub,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vb,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vb,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_aam,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_rho,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_rmean,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advx,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advy,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_drhox,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_drhoy,     kb*jm*im*sizeof(float)));

	//added by mode_internal
	checkCudaErrors(cudaMalloc((void**) &d_q2b,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2lb,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2l,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_uf,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vf,     kb*jm*im*sizeof(float)));

	//added by mode_internal-->profq
	checkCudaErrors(cudaMalloc((void**) &d_kq,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_l,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_t,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_s,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_km,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_kh,     kb*jm*im*sizeof(float)));
	//added by mode_internal-->advt1
	checkCudaErrors(cudaMalloc((void**) &d_tb,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sb,     kb*jm*im*sizeof(float)));

	//added by mode_internal-->bcond(4)
	checkCudaErrors(cudaMalloc((void**) &d_tobw,   kb*jm*nfw*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sobw,   kb*jm*nfw*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tobe,   kb*jm*nfe*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sobe,   kb*jm*nfe*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tobs,   kb*nfs*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sobs,   kb*nfs*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tobn,   kb*nfn*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sobn,   kb*nfn*im*sizeof(float)));

	//added by store_mean
	checkCudaErrors(cudaMalloc((void**) &d_u_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_v_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_w_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_t_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_s_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_rho_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_kh_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_km_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ustks, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vstks, kb*jm*im*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**) &d_xstks_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ystks_mean, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_xstks, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ystks, kb*jm*im*sizeof(float)));


    /////////////////////////////////////////////////////
	//added by momentum3d
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
	checkCudaErrors(cudaMemcpy(d_rmean, rmean, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	//added by mode_internal
	checkCudaErrors(cudaMemcpy(d_q2b, q2b, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2, q2, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2lb, q2lb, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q2l, q2l, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_uf, uf, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vf, vf, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	//added by mode_internal-->profq
	checkCudaErrors(cudaMemcpy(d_kq, kq, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_t, t, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_s, s, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kh, kh, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	//added by mode_internal-->advt1
	checkCudaErrors(cudaMemcpy(d_tb, tb, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sb, sb, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	//added by mode_internal-->bcond(4)
	checkCudaErrors(cudaMemcpy(d_tobw, tobw, kb*jm*nfw*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sobw, sobw, kb*jm*nfw*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tobe, tobe, kb*jm*nfe*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sobe, sobe, kb*jm*nfe*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tobs, tobs, kb*nfs*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sobs, sobs, kb*nfs*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tobn, tobn, kb*nfn*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sobn, sobn, kb*nfn*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	//added by store_mean
	checkCudaErrors(cudaMemcpy(d_ustks, ustks, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vstks, vstks, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_xstks, xstks, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ystks, ystks, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////
	checkCudaErrors(cudaMemset(d_w, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_advx, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_advy, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_drhox, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_drhoy, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_l, 0, kb*jm*im*sizeof(float)));

	checkCudaErrors(cudaMemset(d_u_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_v_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_w_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_t_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_s_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_rho_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_kh_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_km_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_xstks_mean, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ystks_mean, 0, kb*jm*im*sizeof(float)));

	/*
	//below is for global variables memory alloc
	checkCudaErrors(cudaMalloc((void**) &d_t,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_s,	  kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_u,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_v,     kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_ub,    kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_vb,    kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_aam,   kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_rho,   kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advx,  kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_advy,  kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_drhox, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_drhoy, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2b,   kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2,    kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2lb,  kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_q2l,   kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_tb,   kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_sb,   kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_km,	  kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_kh,	  kb*jm*im*sizeof(float)));


	//added by surface forcing
	//in fact we just use s[0] and t[0],
	//what's worse, we multply them by 0...
	//but mode_internal will use whole s and t 
	checkCudaErrors(cudaMemcpy(d_t, t, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_s, s, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));

	//added by surface forcing
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

	//added by surface forcing
	checkCudaErrors(cudaMemcpy(d_rho, rho, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advx, advx, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_advy, advy, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_drhox, drhox, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_drhoy, drhoy, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)

	//added by mode_internal
	checkCudaErrors(cudaMemcpy(d_q2b, q2b, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_q2, q2, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_q2lb, q2lb, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_q2l, q2l, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_tb, tb, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_sb, sb, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_km, km, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)
	checkCudaErrors(cudaMemcpy(d_kh, kh, kb*jm*im*sizeof(float), 
				cudaMemcpyHostToDevice));//it is assigned in initial.f(baropg_f)

	*/
	
	return;
}


void init_cuda_local(){


	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp1, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp2, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp3, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp4, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp5, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_3d_tmp6, kb*jm*im*sizeof(float)));

	//checkCudaErrors(cudaMalloc((void**) &d_3d_tmp7, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**) &d_3d_tmp8, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**) &d_3d_tmp9, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**) &d_3d_tmp10, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**) &d_3d_tmp11, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**) &d_3d_tmp12, kb*jm*im*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp1, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp2, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp3, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp4, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp5, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp6, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp7, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp8, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp9, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp10, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp11, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp12, jm*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_tmp13, jm*im*sizeof(float)));

	//for data exchange in exchange2d_mpi
	checkCudaErrors(cudaMalloc((void**) &d_1d_ny_tmp0, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_1d_ny_tmp1, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_1d_ny_tmp2, jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_1d_ny_tmp3, jm*sizeof(float)));

	//for data exchange in exchange3d_mpi
	checkCudaErrors(cudaMalloc((void**) &d_2d_ny_nz_tmp0, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_ny_nz_tmp1, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_ny_nz_tmp2, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_ny_nz_tmp3, kb*jm*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**) &d_2d_nx_nz_tmp0, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_nx_nz_tmp1, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_nx_nz_tmp2, kb*im*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**) &d_2d_nx_nz_tmp3, kb*im*sizeof(float)));

	checkCudaErrors(cudaStreamCreate(&stream[0]));	
	for (int i = 1; i < 5; i++){
		checkCudaErrors(cudaStreamCreateWithPriority(
						&stream[i], cudaStreamDefault, -1));	
	}

	checkCudaErrors(cudaMemset(d_3d_tmp0, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_3d_tmp1, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_3d_tmp2, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_3d_tmp3, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_3d_tmp4, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_3d_tmp5, 0, kb*jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_3d_tmp6, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_3d_tmp7, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_3d_tmp8, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_3d_tmp9, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_3d_tmp10, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_3d_tmp11, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_3d_tmp12, 0, kb*jm*im*sizeof(float)));

	checkCudaErrors(cudaMemset(d_2d_tmp0, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp1, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp2, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp3, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp4, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp5, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp6, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp7, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp8, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp9, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp10, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp11, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp12, 0, jm*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_tmp13, 0, jm*im*sizeof(float)));


	checkCudaErrors(cudaMemset(d_1d_ny_tmp0, 0, jm*sizeof(float)));
	checkCudaErrors(cudaMemset(d_1d_ny_tmp1, 0, jm*sizeof(float)));
	checkCudaErrors(cudaMemset(d_1d_ny_tmp2, 0, jm*sizeof(float)));
	checkCudaErrors(cudaMemset(d_1d_ny_tmp3, 0, jm*sizeof(float)));

	checkCudaErrors(cudaMemset(d_2d_ny_nz_tmp0, 0, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_ny_nz_tmp1, 0, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_ny_nz_tmp2, 0, kb*jm*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_ny_nz_tmp3, 0, kb*jm*sizeof(float)));

	checkCudaErrors(cudaMemset(d_2d_nx_nz_tmp0, 0, kb*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_nx_nz_tmp1, 0, kb*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_nx_nz_tmp2, 0, kb*im*sizeof(float)));
	checkCudaErrors(cudaMemset(d_2d_nx_nz_tmp3, 0, kb*im*sizeof(float)));


	//checkCudaErrors(cudaMemset(d_3d_tmp0_d, 0, kb*jm*im*sizeof(double)));
	//checkCudaErrors(cudaMemset(d_3d_tmp1_d, 0, kb*jm*im*sizeof(double)));
	//checkCudaErrors(cudaMemset(d_3d_tmp2_d, 0, kb*jm*im*sizeof(double)));
	//checkCudaErrors(cudaMemset(d_3d_tmp3_d, 0, kb*jm*im*sizeof(double)));

	//checkCudaErrors(cudaMemset(d_uf, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_vf, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_w, 0, kb*jm*im*sizeof(float)));
	//checkCudaErrors(cudaMemset(d_l, 0, kb*jm*im*sizeof(float)));

	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	return;
}


//void init_cuda_ipc(){
void exchangeMemHandle(){
	checkCudaErrors(cudaIpcGetMemHandle(&handle_ctsurf, (void*)d_ctsurf));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_ctbot, (void*)d_ctbot));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_celg, (void*)d_celg));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_cjbar, (void*)d_cjbar));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_cadv, (void*)d_cadv));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_cpvf, (void*)d_cpvf));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_cten, (void*)d_cten));


	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp0, (void*)d_2d_tmp0));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp1, (void*)d_2d_tmp1));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp2, (void*)d_2d_tmp2));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp3, (void*)d_2d_tmp3));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp4, (void*)d_2d_tmp4));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp5, (void*)d_2d_tmp5));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp6, (void*)d_2d_tmp6));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp7, (void*)d_2d_tmp7));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp8, (void*)d_2d_tmp8));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp9, (void*)d_2d_tmp9));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp10, (void*)d_2d_tmp10));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp11, (void*)d_2d_tmp11));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp12, (void*)d_2d_tmp12));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_2d_tmp13, (void*)d_2d_tmp13));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_totx, (void*)d_totx));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_toty, (void*)d_toty));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_3d_tmp0, (void*)d_3d_tmp0));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_3d_tmp1, (void*)d_3d_tmp1));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_3d_tmp2, (void*)d_3d_tmp2));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_wubot, (void*)d_wubot));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_wvbot, (void*)d_wvbot));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_aam, (void*)d_aam));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_elf, (void*)d_elf));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_uaf, (void*)d_uaf));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_vaf, (void*)d_vaf));

	checkCudaErrors(cudaIpcGetMemHandle(&handle_w, (void*)d_w));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_uf, (void*)d_uf));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_vf, (void*)d_vf));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_kh, (void*)d_kh));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_km, (void*)d_km));
	checkCudaErrors(cudaIpcGetMemHandle(&handle_kq, (void*)d_kq));

	MPI_Status status;

	if (n_west != -1){
		MPI_Send(&handle_ctsurf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_ctbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_celg, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_cjbar, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_cadv, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_cpvf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_cten, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);


		MPI_Send(&handle_2d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp3, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp4, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp5, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp6, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp7, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp8, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp9, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp10, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp11, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp12, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_2d_tmp13, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);


		MPI_Send(&handle_totx, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_toty, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		MPI_Send(&handle_3d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_3d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_3d_tmp2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		MPI_Send(&handle_wubot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_wvbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		MPI_Send(&handle_aam, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		MPI_Send(&handle_elf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		
		MPI_Send(&handle_uaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_vaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		MPI_Send(&handle_w, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		MPI_Send(&handle_uf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);
		MPI_Send(&handle_vf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_west, 0, pom_comm);

		//////////////////////////////////////////////////////////////

		MPI_Recv(&handle_ctsurf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_ctbot_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_celg_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_cjbar_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_cadv_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_cpvf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_cten_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);


		MPI_Recv(&handle_2d_tmp0_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp1_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp2_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp3_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp4_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp5_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp6_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp7_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp8_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp9_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp10_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp11_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp12_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp13_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);


		MPI_Recv(&handle_totx_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_toty_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);


		MPI_Recv(&handle_3d_tmp0_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_3d_tmp1_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_3d_tmp2_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);

		MPI_Recv(&handle_wubot_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_wvbot_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);

		MPI_Recv(&handle_aam_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);

		MPI_Recv(&handle_elf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);

		MPI_Recv(&handle_uaf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_vaf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);

		MPI_Recv(&handle_w_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);

		MPI_Recv(&handle_uf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
		MPI_Recv(&handle_vf_west, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_west, 0, pom_comm, &status);
	}


	if (n_east != -1){
		MPI_Recv(&handle_ctsurf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_ctbot_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_celg_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_cjbar_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_cadv_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_cpvf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_cten_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	


		MPI_Recv(&handle_2d_tmp0_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp1_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp2_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp3_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp4_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp5_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp6_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp7_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp8_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp9_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp10_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp11_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp12_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_2d_tmp13_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_totx_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_toty_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_3d_tmp0_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_3d_tmp1_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_3d_tmp2_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_wubot_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_wvbot_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_aam_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_elf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_uaf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_vaf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_w_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	

		MPI_Recv(&handle_uf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	
		MPI_Recv(&handle_vf_east, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_east, 0, pom_comm, &status);	


		MPI_Send(&handle_ctsurf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_ctbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_celg, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_cjbar, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_cadv, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_cpvf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_cten, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);



		MPI_Send(&handle_2d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp3, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp4, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp5, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp6, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp7, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp8, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp9, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp10, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp11, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp12, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_2d_tmp13, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);


		MPI_Send(&handle_totx, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_toty, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);


		MPI_Send(&handle_3d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_3d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_3d_tmp2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

		MPI_Send(&handle_wubot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_wvbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

		MPI_Send(&handle_aam, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

		MPI_Send(&handle_elf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

		MPI_Send(&handle_uaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_vaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

		MPI_Send(&handle_w, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

		MPI_Send(&handle_uf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);
		MPI_Send(&handle_vf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_east, 0, pom_comm);

	}

	if (n_south != -1){

		MPI_Send(&handle_3d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		MPI_Send(&handle_3d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		MPI_Send(&handle_3d_tmp2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		MPI_Send(&handle_2d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		MPI_Send(&handle_2d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		MPI_Send(&handle_wubot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		MPI_Send(&handle_wvbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		
		MPI_Send(&handle_aam, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		MPI_Send(&handle_elf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		MPI_Send(&handle_uaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		MPI_Send(&handle_vaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		MPI_Send(&handle_w, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		MPI_Send(&handle_uf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);
		MPI_Send(&handle_vf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_south, 0, pom_comm);

		////////////////////////////////////////////////////////////////////

		MPI_Recv(&handle_3d_tmp0_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	
		MPI_Recv(&handle_3d_tmp1_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	
		MPI_Recv(&handle_3d_tmp2_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	

		MPI_Recv(&handle_2d_tmp0_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp1_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);

		MPI_Recv(&handle_wubot_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	
		MPI_Recv(&handle_wvbot_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	

		MPI_Recv(&handle_aam_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);

		MPI_Recv(&handle_elf_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	

		MPI_Recv(&handle_uaf_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);
		MPI_Recv(&handle_vaf_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);

		MPI_Recv(&handle_w_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	

		MPI_Recv(&handle_uf_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	
		MPI_Recv(&handle_vf_south, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_south, 0, pom_comm, &status);	
	}

	if (n_north != -1){

		MPI_Recv(&handle_3d_tmp0_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	
		MPI_Recv(&handle_3d_tmp1_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	
		MPI_Recv(&handle_3d_tmp2_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	

		MPI_Recv(&handle_2d_tmp0_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);
		MPI_Recv(&handle_2d_tmp1_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);
		
		MPI_Recv(&handle_wubot_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	
		MPI_Recv(&handle_wvbot_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	

		MPI_Recv(&handle_aam_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);

		MPI_Recv(&handle_elf_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	

		MPI_Recv(&handle_uaf_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);
		MPI_Recv(&handle_vaf_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);

		MPI_Recv(&handle_w_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	

		MPI_Recv(&handle_uf_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	
		MPI_Recv(&handle_vf_north, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
				 n_north, 0, pom_comm, &status);	

		////////////////////////////////////////////////////////////////////

		MPI_Send(&handle_3d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
		MPI_Send(&handle_3d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
		MPI_Send(&handle_3d_tmp2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_2d_tmp0, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
		MPI_Send(&handle_2d_tmp1, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_wubot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
		MPI_Send(&handle_wvbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_aam, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_elf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_uaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
		MPI_Send(&handle_vaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_w, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);

		MPI_Send(&handle_uf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
		MPI_Send(&handle_vf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
				 n_north, 0, pom_comm);
	}

	if (!(n_east == -1 && n_west == -1)){
		int nproc_x = (im_global-2)/(im_local-2);
		printf("xsz_debug: nproc_x = %d, rank:%d\n", nproc_x, my_task);
		if (n_west == -1){
			MPI_Send(&handle_elf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_elf_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_uaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_uaf_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_vaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_vaf_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_w, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_w_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_uf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_uf_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_vf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_vf_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_kh, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_kh_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_km, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_km_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_kq, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_kq_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_wubot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_wubot_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

			MPI_Send(&handle_wvbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task+nproc_x-1, 0, pom_comm);
			MPI_Recv(&handle_wvbot_east_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task+nproc_x-1, 0, pom_comm, &status);

		}

		if (n_east == -1){
			MPI_Recv(&handle_elf_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_elf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_uaf_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_uaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_vaf_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_vaf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_w_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_w, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_uf_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_uf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_vf_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_vf, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_kh_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_kh, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_km_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_km, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_kq_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_kq, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_wubot_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_wubot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

			MPI_Recv(&handle_wvbot_west_most, sizeof(cudaIpcMemHandle_t), 
					 MPI_BYTE, my_task-nproc_x+1, 0, pom_comm, &status);

			MPI_Send(&handle_wvbot, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
					 my_task-nproc_x+1, 0, pom_comm);

		}

		//if (n_west == -1){
		//	int access;
		//	checkCudaErrors(cudaDeviceCanAccessPeer(&access, my_task, 
		//											n_east));
		//	if (access){
		//		checkCudaErrors(cudaDeviceEnablePeerAccess(n_east, 0));	
		//		printf("Enable Peer Access from %d to %d\n", 
		//			   my_task, n_east);
		//		checkCudaErrors(cudaSetDevice(n_east));
		//		checkCudaErrors(cudaDeviceEnablePeerAccess(my_task, 0));
		//		printf("Enable Peer Access from %d to %d\n", 
		//			   n_east, my_task);
		//		checkCudaErrors(cudaSetDevice(my_task));
		//	}
		//}

		//if (n_east == -1){
		//	int access;	
		//	checkCudaErrors(cudaDeviceCanAccessPeer(&access, my_task, 
		//											n_west));
		//	if (access){
		//		checkCudaErrors(cudaDeviceEnablePeerAccess(n_west, 0));	
		//		printf("Enable Peer Access from %d to %d\n", 
		//			   my_task, n_west);
		//		checkCudaErrors(cudaSetDevice(n_west));
		//		checkCudaErrors(cudaDeviceEnablePeerAccess(my_task, 0));
		//		printf("Enable Peer Access from %d to %d\n", 
		//			   n_west, my_task);
		//		checkCudaErrors(cudaSetDevice(my_task));
		//	}
		//}
	}

	printf("Exchange Memory Handle End!\n");
}

void openMemHandle(){

	printf("This program is now hard-coded for 1, 2, or 4 process\n");
	printf("For 2 and 4 process, There are 2 processes distributed in longitude\n");

	if (n_east != -1){
		//checkCudaErrors(cudaSetDevice(n_east));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_ctsurf_east, handle_ctsurf_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_ctbot_east, handle_ctbot_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_celg_east, handle_celg_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cjbar_east, handle_cjbar_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cadv_east, handle_cadv_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cpvf_east, handle_cpvf_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cten_east, handle_cten_east,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp0_east, handle_2d_tmp0_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp1_east, handle_2d_tmp1_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp2_east, handle_2d_tmp2_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp3_east, handle_2d_tmp3_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp4_east, handle_2d_tmp4_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp5_east, handle_2d_tmp5_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp6_east, handle_2d_tmp6_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp7_east, handle_2d_tmp7_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp8_east, handle_2d_tmp8_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp9_east, handle_2d_tmp9_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp10_east, handle_2d_tmp10_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp11_east, handle_2d_tmp11_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp12_east, handle_2d_tmp12_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp13_east, handle_2d_tmp13_east,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_totx_east, handle_totx_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_toty_east, handle_toty_east,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp0_east, handle_3d_tmp0_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp1_east, handle_3d_tmp1_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp2_east, handle_3d_tmp2_east,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wubot_east, handle_wubot_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wvbot_east, handle_wvbot_east,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_aam_east, handle_aam_east,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_elf_east, handle_elf_east,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uaf_east, handle_uaf_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vaf_east, handle_vaf_east,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_w_east, handle_w_east,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uf_east, handle_uf_east,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vf_east, handle_vf_east,
							 cudaIpcMemLazyEnablePeerAccess));

		//checkCudaErrors(cudaSetDevice(my_task));
	}

	if (n_west != -1){

		//checkCudaErrors(cudaSetDevice(n_west));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_ctsurf_west, handle_ctsurf_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_ctbot_west, handle_ctbot_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_celg_west, handle_celg_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cjbar_west, handle_cjbar_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cadv_west, handle_cadv_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cpvf_west, handle_cpvf_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_cten_west, handle_cten_west,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp0_west, handle_2d_tmp0_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp1_west, handle_2d_tmp1_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp2_west, handle_2d_tmp2_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp3_west, handle_2d_tmp3_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp4_west, handle_2d_tmp4_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp5_west, handle_2d_tmp5_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp6_west, handle_2d_tmp6_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp7_west, handle_2d_tmp7_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp8_west, handle_2d_tmp8_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp9_west, handle_2d_tmp9_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp10_west, handle_2d_tmp10_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp11_west, handle_2d_tmp11_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp12_west, handle_2d_tmp12_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp13_west, handle_2d_tmp13_west,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_totx_west, handle_totx_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_toty_west, handle_toty_west,
							 cudaIpcMemLazyEnablePeerAccess));


		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp0_west, handle_3d_tmp0_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp1_west, handle_3d_tmp1_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp2_west, handle_3d_tmp2_west,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wubot_west, handle_wubot_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wvbot_west, handle_wvbot_west,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_aam_west, handle_aam_west,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_elf_west, handle_elf_west,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uaf_west, handle_uaf_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vaf_west, handle_vaf_west,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_w_west, handle_w_west,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uf_west, handle_uf_west,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vf_west, handle_vf_west,
							 cudaIpcMemLazyEnablePeerAccess));

		//checkCudaErrors(cudaSetDevice(my_task));
	}

	if (n_south != -1){
		checkCudaErrors(cudaSetDevice(n_south));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp0_south, handle_3d_tmp0_south,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp1_south, handle_3d_tmp1_south,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp2_south, handle_3d_tmp2_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp0_south, handle_2d_tmp0_south,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp1_south, handle_2d_tmp1_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wubot_south, handle_wubot_south,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wvbot_south, handle_wvbot_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_aam_south, handle_aam_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_elf_south, handle_elf_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uaf_south, handle_uaf_south,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vaf_south, handle_vaf_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_w_south, handle_w_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uf_south, handle_uf_south,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vf_south, handle_vf_south,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaSetDevice(my_task));
	}

	if (n_north != -1){
		checkCudaErrors(cudaSetDevice(n_north));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp0_north, handle_3d_tmp0_north,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp1_north, handle_3d_tmp1_north,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_3d_tmp2_north, handle_3d_tmp2_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp0_north, handle_2d_tmp0_north,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_2d_tmp1_north, handle_2d_tmp1_north,
							 cudaIpcMemLazyEnablePeerAccess));
	
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wubot_north, handle_wubot_north,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_wvbot_north, handle_wvbot_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_aam_north, handle_aam_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_elf_north, handle_elf_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uaf_north, handle_uaf_north,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vaf_north, handle_vaf_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_w_north, handle_w_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_uf_north, handle_uf_north,
							 cudaIpcMemLazyEnablePeerAccess));
		checkCudaErrors(cudaIpcOpenMemHandle(
							(void**)&d_vf_north, handle_vf_north,
							 cudaIpcMemLazyEnablePeerAccess));

		checkCudaErrors(cudaSetDevice(my_task));
	}

	if (!(n_east == -1 && n_west == -1)){
		int nproc_x = (im_global-2)/(im_local-2);
		if (nproc_x == 2){
			if (n_west == -1){
				d_elf_east_most = d_elf_east;
				d_uaf_east_most = d_uaf_east;
				d_vaf_east_most = d_vaf_east;
				d_w_east_most = d_w_east;
				d_uf_east_most = d_uf_east;
				d_vf_east_most = d_vf_east;
				//d_kh_east_most = d_kh_east;
				//d_km_east_most = d_km_east;
				//d_kq_east_most = d_kq_east;
				d_wubot_east_most = d_wubot_east;
				d_wvbot_east_most = d_wvbot_east;

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kh_east_most, 
								handle_kh_east_most,
								cudaIpcMemLazyEnablePeerAccess));
				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_km_east_most, 
								handle_km_east_most,
								cudaIpcMemLazyEnablePeerAccess));
				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kq_east_most, 
								handle_kq_east_most,
								cudaIpcMemLazyEnablePeerAccess));
			}
			if (n_east == -1){
				d_elf_west_most = d_elf_west;
				d_uaf_west_most = d_uaf_west;
				d_vaf_west_most = d_vaf_west;
				d_w_west_most = d_w_west;
				d_uf_west_most = d_uf_west;
				d_vf_west_most = d_vf_west;
				//d_kh_west_most = d_kh_west;
				//d_km_west_most = d_km_west;
				//d_kq_west_most = d_kq_west;
				d_wubot_west_most = d_wubot_west;
				d_wvbot_west_most = d_wvbot_west;

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kh_west_most, 
								handle_kh_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_km_west_most, 
								handle_km_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kq_west_most, 
								handle_kq_west_most,
								cudaIpcMemLazyEnablePeerAccess));
			}
		}else{

			if (n_west == -1){
				//checkCudaErrors(cudaSetDevice(my_task+nproc_x-1));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_elf_east_most, handle_elf_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_uaf_east_most, handle_uaf_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_vaf_east_most, handle_vaf_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_w_east_most, handle_w_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_uf_east_most, handle_uf_east_most,
								cudaIpcMemLazyEnablePeerAccess));
				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_vf_east_most, handle_vf_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kh_east_most, handle_kh_east_most,
								cudaIpcMemLazyEnablePeerAccess));
				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_km_east_most, handle_km_east_most,
								cudaIpcMemLazyEnablePeerAccess));
				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kq_east_most, handle_kq_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_wubot_east_most, 
								handle_wubot_east_most,
								cudaIpcMemLazyEnablePeerAccess));
				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_wvbot_east_most, 
								handle_wvbot_east_most,
								cudaIpcMemLazyEnablePeerAccess));

				//checkCudaErrors(cudaSetDevice(my_task));

			}

			if (n_east == -1){
				//checkCudaErrors(cudaSetDevice(my_task-(nproc_x-1)));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_elf_west_most, handle_elf_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_uaf_west_most, handle_uaf_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_vaf_west_most, handle_vaf_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_w_west_most, handle_w_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_uf_west_most, handle_uf_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_vf_west_most, handle_vf_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kh_west_most, handle_kh_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_km_west_most, handle_km_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_kq_west_most, handle_kq_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_wubot_west_most, 
								handle_wubot_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				checkCudaErrors(cudaIpcOpenMemHandle(
								(void**)&d_wvbot_west_most, 
								handle_wvbot_west_most,
								cudaIpcMemLazyEnablePeerAccess));

				//checkCudaErrors(cudaSetDevice(my_task));
			}
		}
	}
}

void init_cuda_peer(){

	///////////////////////////////////////////////////
	//This function is of no use, just to warm the hardware
	//we find a 1s latency when first calling cudaMemcpyPeer
	//so we call it first in init phase
	//I think it is relevant to JIT
	
	cudaMemcpy3DPeerParms p_east_recv={0};
	p_east_recv.extent = make_cudaExtent(sizeof(float), jm, kb);
	p_east_recv.dstDevice = n_east;
	p_east_recv.dstPtr = make_cudaPitchedPtr(d_3d_tmp0, im*sizeof(float), im, jm);
	p_east_recv.srcDevice = my_task;
	p_east_recv.srcPtr = make_cudaPitchedPtr(d_3d_tmp1, im*sizeof(float), im, jm);

	cudaMemcpy3DPeerParms p_west_recv={0};
	p_west_recv.extent = make_cudaExtent(sizeof(float), jm, kb);
	p_west_recv.dstDevice = n_west;
	p_west_recv.dstPtr = make_cudaPitchedPtr(d_3d_tmp0, im*sizeof(float), im, jm);
	p_west_recv.srcDevice = my_task;
	p_west_recv.srcPtr = make_cudaPitchedPtr(d_3d_tmp1, im*sizeof(float), im, jm);

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy3DPeer(&p_east_recv));	
	}

	if (n_west != -1){
		checkCudaErrors(cudaMemcpy3DPeer(&p_west_recv));	
	}
}

void init_cuda_pinned_memory(){
	//mpi
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_nx_tmp0, 
								  im*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_nx_tmp1, 
								  im*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_nx_tmp2, 
								  im*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_nx_tmp3, 
								  im*sizeof(float), 
								  cudaHostAllocPortable));

	checkCudaErrors(cudaHostAlloc((void**)&h_1d_ny_tmp0, 
								  jm*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_ny_tmp1, 
								  jm*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_ny_tmp2, 
								  jm*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_1d_ny_tmp3, 
								  jm*sizeof(float), 
								  cudaHostAllocPortable));

	checkCudaErrors(cudaHostAlloc((void**)&h_2d_nx_nz_tmp0, 
								  im*kb*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_2d_nx_nz_tmp1, 
								  im*kb*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_2d_nx_nz_tmp2, 
								  im*kb*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_2d_nx_nz_tmp3, 
								  im*kb*sizeof(float), 
								  cudaHostAllocPortable));

	checkCudaErrors(cudaHostAlloc((void**)&h_2d_ny_nz_tmp0, 
								  jm*kb*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_2d_ny_nz_tmp1, 
								  jm*kb*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_2d_ny_nz_tmp2, 
								  jm*kb*sizeof(float), 
								  cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&h_2d_ny_nz_tmp3, 
								  jm*kb*sizeof(float), 
								  cudaHostAllocPortable));

	//print_section
	checkCudaErrors(cudaHostRegister(tb, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(sb, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(ub, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(vb, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));


	checkCudaErrors(cudaHostRegister(dt, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(et, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));

	//output_copy_back
	checkCudaErrors(cudaHostRegister(u, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(v, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(w, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(t, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(s, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(rho, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(kh, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(km, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));

	checkCudaErrors(cudaHostRegister(uab, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(vab, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(elb, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));

	checkCudaErrors(cudaHostRegister(u_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(v_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(w_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(t_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(s_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(rho_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(kh_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(km_mean, kb*jm*im*sizeof(float), 
									 cudaHostRegisterPortable));

	checkCudaErrors(cudaHostRegister(uab_mean, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(vab_mean, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(elb_mean, jm*im*sizeof(float), 
									 cudaHostRegisterPortable));

	return;
}

void finalize_cuda_gpu(){
	//fclose(out);

	//1D const 
	checkCudaErrors(cudaFree(d_zz));
	checkCudaErrors(cudaFree(d_dz));
	checkCudaErrors(cudaFree(d_uabe));
	checkCudaErrors(cudaFree(d_uabw));
	checkCudaErrors(cudaFree(d_ele));
	checkCudaErrors(cudaFree(d_elw));
	checkCudaErrors(cudaFree(d_vabs));
	checkCudaErrors(cudaFree(d_vabn));
	checkCudaErrors(cudaFree(d_els));
	checkCudaErrors(cudaFree(d_eln));
	checkCudaErrors(cudaFree(d_dzz));
	checkCudaErrors(cudaFree(d_z));


	//checkCudaErrors(cudaFree(d_j_global));
	//checkCudaErrors(cudaFree(d_aam_aid));


	//2D const
	checkCudaErrors(cudaFree(d_dx));
	checkCudaErrors(cudaFree(d_dy));
	checkCudaErrors(cudaFree(d_aru));
	checkCudaErrors(cudaFree(d_arv));
	checkCudaErrors(cudaFree(d_aamfrz));
	checkCudaErrors(cudaFree(d_art));
	checkCudaErrors(cudaFree(d_cor));
	checkCudaErrors(cudaFree(d_h));
	checkCudaErrors(cudaFree(d_fsm));
	checkCudaErrors(cudaFree(d_frz));


	//3D const
	checkCudaErrors(cudaFree(d_tclim));
	checkCudaErrors(cudaFree(d_sclim));

	checkCudaErrors(cudaFree(d_relax_aid));



	//2D
	checkCudaErrors(cudaFree(d_vfluxf));
	checkCudaErrors(cudaFree(d_e_atmos));
	checkCudaErrors(cudaFree(d_swrad));
	checkCudaErrors(cudaFree(d_dum));
	checkCudaErrors(cudaFree(d_dvm));
	checkCudaErrors(cudaFree(d_dt));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(d_adx2d));
	checkCudaErrors(cudaFree(d_ady2d));
	checkCudaErrors(cudaFree(d_drx2d));
	checkCudaErrors(cudaFree(d_dry2d));
	checkCudaErrors(cudaFree(d_aam2d));
	checkCudaErrors(cudaFree(d_ua));
	checkCudaErrors(cudaFree(d_va));
	checkCudaErrors(cudaFree(d_uab));
	checkCudaErrors(cudaFree(d_vab));
	checkCudaErrors(cudaFree(d_wubot));
	checkCudaErrors(cudaFree(d_wvbot));
	checkCudaErrors(cudaFree(d_advua));
	checkCudaErrors(cudaFree(d_advva));
	checkCudaErrors(cudaFree(d_el));
	checkCudaErrors(cudaFree(d_egf));
	checkCudaErrors(cudaFree(d_utf));
	checkCudaErrors(cudaFree(d_vtf));
	checkCudaErrors(cudaFree(d_cbc));


	checkCudaErrors(cudaFree(d_elb));
	checkCudaErrors(cudaFree(d_elf));
	checkCudaErrors(cudaFree(d_wusurf));
	checkCudaErrors(cudaFree(d_wvsurf));
	checkCudaErrors(cudaFree(d_etf));
	checkCudaErrors(cudaFree(d_uaf));
	checkCudaErrors(cudaFree(d_vaf));


	checkCudaErrors(cudaFree(d_utb));
	checkCudaErrors(cudaFree(d_vtb));
	checkCudaErrors(cudaFree(d_etb));
	checkCudaErrors(cudaFree(d_vfluxb));
	checkCudaErrors(cudaFree(d_tsurf));
	checkCudaErrors(cudaFree(d_wtsurf));
	checkCudaErrors(cudaFree(d_wssurf));
	checkCudaErrors(cudaFree(d_ssurf));


	checkCudaErrors(cudaFree(d_tbe));
	checkCudaErrors(cudaFree(d_sbe));
	checkCudaErrors(cudaFree(d_tbw));
	checkCudaErrors(cudaFree(d_sbw));
	checkCudaErrors(cudaFree(d_tbs));
	checkCudaErrors(cudaFree(d_sbs));
	checkCudaErrors(cudaFree(d_tbn));
	checkCudaErrors(cudaFree(d_sbn));


	checkCudaErrors(cudaFree(d_egb));
	checkCudaErrors(cudaFree(d_et));

	checkCudaErrors(cudaFree(d_uab_mean));
	checkCudaErrors(cudaFree(d_vab_mean));
	checkCudaErrors(cudaFree(d_elb_mean));
	checkCudaErrors(cudaFree(d_wusurf_mean));
	checkCudaErrors(cudaFree(d_wvsurf_mean));
	checkCudaErrors(cudaFree(d_wtsurf_mean));
	checkCudaErrors(cudaFree(d_wssurf_mean));
	checkCudaErrors(cudaFree(d_usrf_mean));
	checkCudaErrors(cudaFree(d_vsrf_mean));
	checkCudaErrors(cudaFree(d_elsrf_mean));
	checkCudaErrors(cudaFree(d_uwsrf_mean));
	checkCudaErrors(cudaFree(d_vwsrf_mean));
	checkCudaErrors(cudaFree(d_uwsrf));
	checkCudaErrors(cudaFree(d_vwsrf));
	checkCudaErrors(cudaFree(d_utf_mean));
	checkCudaErrors(cudaFree(d_vtf_mean));
	checkCudaErrors(cudaFree(d_celg_mean));
	checkCudaErrors(cudaFree(d_ctsurf_mean));
	checkCudaErrors(cudaFree(d_celg));
	checkCudaErrors(cudaFree(d_ctsurf));
	checkCudaErrors(cudaFree(d_cpvf_mean));
	checkCudaErrors(cudaFree(d_cjbar_mean));
	checkCudaErrors(cudaFree(d_cpvf));
	checkCudaErrors(cudaFree(d_cjbar));
	checkCudaErrors(cudaFree(d_cadv_mean));
	checkCudaErrors(cudaFree(d_cten_mean));
	checkCudaErrors(cudaFree(d_cadv));
	checkCudaErrors(cudaFree(d_cten));
	checkCudaErrors(cudaFree(d_ctbot_mean));
	checkCudaErrors(cudaFree(d_ctbot));


	//3D
	checkCudaErrors(cudaFree(d_w));
	checkCudaErrors(cudaFree(d_u));
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_ub));
	checkCudaErrors(cudaFree(d_vb));
	checkCudaErrors(cudaFree(d_aam));
	checkCudaErrors(cudaFree(d_rho));
	checkCudaErrors(cudaFree(d_rmean));
	checkCudaErrors(cudaFree(d_advx));
	checkCudaErrors(cudaFree(d_advy));
	checkCudaErrors(cudaFree(d_drhox));
	checkCudaErrors(cudaFree(d_drhoy));


	checkCudaErrors(cudaFree(d_q2b));
	checkCudaErrors(cudaFree(d_q2));
	checkCudaErrors(cudaFree(d_q2lb));
	checkCudaErrors(cudaFree(d_q2l));
	checkCudaErrors(cudaFree(d_uf));
	checkCudaErrors(cudaFree(d_vf));
	checkCudaErrors(cudaFree(d_kq));
	checkCudaErrors(cudaFree(d_l));
	checkCudaErrors(cudaFree(d_t));
	checkCudaErrors(cudaFree(d_s));
	checkCudaErrors(cudaFree(d_km));
	checkCudaErrors(cudaFree(d_kh));
	checkCudaErrors(cudaFree(d_tb));
	checkCudaErrors(cudaFree(d_sb));

	checkCudaErrors(cudaFree(d_tobw));
	checkCudaErrors(cudaFree(d_sobw));
	checkCudaErrors(cudaFree(d_tobe));
	checkCudaErrors(cudaFree(d_sobe));
	checkCudaErrors(cudaFree(d_tobs));
	checkCudaErrors(cudaFree(d_sobs));
	checkCudaErrors(cudaFree(d_tobn));
	checkCudaErrors(cudaFree(d_sobn));


	checkCudaErrors(cudaFree(d_u_mean));
	checkCudaErrors(cudaFree(d_v_mean));
	checkCudaErrors(cudaFree(d_w_mean));
	checkCudaErrors(cudaFree(d_t_mean));
	checkCudaErrors(cudaFree(d_s_mean));
	checkCudaErrors(cudaFree(d_rho_mean));
	checkCudaErrors(cudaFree(d_kh_mean));
	checkCudaErrors(cudaFree(d_km_mean));
	checkCudaErrors(cudaFree(d_ustks));
	checkCudaErrors(cudaFree(d_vstks));

	checkCudaErrors(cudaFree(d_xstks_mean));
	checkCudaErrors(cudaFree(d_ystks_mean));
	checkCudaErrors(cudaFree(d_xstks));
	checkCudaErrors(cudaFree(d_ystks));

	
	checkCudaErrors(cudaFree(d_3d_tmp0));
	checkCudaErrors(cudaFree(d_3d_tmp1));
	checkCudaErrors(cudaFree(d_3d_tmp2));
	checkCudaErrors(cudaFree(d_3d_tmp3));
	checkCudaErrors(cudaFree(d_3d_tmp4));
	checkCudaErrors(cudaFree(d_3d_tmp5));
	checkCudaErrors(cudaFree(d_3d_tmp6));
	//checkCudaErrors(cudaFree(d_3d_tmp7));
	//checkCudaErrors(cudaFree(d_3d_tmp8));
	//checkCudaErrors(cudaFree(d_3d_tmp9));
	//checkCudaErrors(cudaFree(d_3d_tmp10));
	//checkCudaErrors(cudaFree(d_3d_tmp11));
	//checkCudaErrors(cudaFree(d_3d_tmp12));


	checkCudaErrors(cudaFree(d_2d_tmp0));
	checkCudaErrors(cudaFree(d_2d_tmp1));
	checkCudaErrors(cudaFree(d_2d_tmp2));
	checkCudaErrors(cudaFree(d_2d_tmp3));
	checkCudaErrors(cudaFree(d_2d_tmp4));
	checkCudaErrors(cudaFree(d_2d_tmp5));
	checkCudaErrors(cudaFree(d_2d_tmp6));
	checkCudaErrors(cudaFree(d_2d_tmp7));
	checkCudaErrors(cudaFree(d_2d_tmp8));
	checkCudaErrors(cudaFree(d_2d_tmp9));
	checkCudaErrors(cudaFree(d_2d_tmp10));
	checkCudaErrors(cudaFree(d_2d_tmp11));
	checkCudaErrors(cudaFree(d_2d_tmp12));
	checkCudaErrors(cudaFree(d_2d_tmp13));

	checkCudaErrors(cudaFree(d_1d_ny_tmp0));
	checkCudaErrors(cudaFree(d_1d_ny_tmp1));
	checkCudaErrors(cudaFree(d_1d_ny_tmp2));
	checkCudaErrors(cudaFree(d_1d_ny_tmp3));

	checkCudaErrors(cudaFree(d_2d_ny_nz_tmp0));
	checkCudaErrors(cudaFree(d_2d_ny_nz_tmp1));
	checkCudaErrors(cudaFree(d_2d_ny_nz_tmp2));
	checkCudaErrors(cudaFree(d_2d_ny_nz_tmp3));

	checkCudaErrors(cudaFree(d_2d_nx_nz_tmp0));
	checkCudaErrors(cudaFree(d_2d_nx_nz_tmp1));
	checkCudaErrors(cudaFree(d_2d_nx_nz_tmp2));
	checkCudaErrors(cudaFree(d_2d_nx_nz_tmp3));

	//free host alloc
	checkCudaErrors(cudaFreeHost(h_1d_nx_tmp0));
	checkCudaErrors(cudaFreeHost(h_1d_nx_tmp1));
	checkCudaErrors(cudaFreeHost(h_1d_nx_tmp2));
	checkCudaErrors(cudaFreeHost(h_1d_nx_tmp3));

	checkCudaErrors(cudaFreeHost(h_1d_ny_tmp0));
	checkCudaErrors(cudaFreeHost(h_1d_ny_tmp1));
	checkCudaErrors(cudaFreeHost(h_1d_ny_tmp2));
	checkCudaErrors(cudaFreeHost(h_1d_ny_tmp3));

	checkCudaErrors(cudaFreeHost(h_2d_nx_nz_tmp0));
	checkCudaErrors(cudaFreeHost(h_2d_nx_nz_tmp1));
	checkCudaErrors(cudaFreeHost(h_2d_nx_nz_tmp2));
	checkCudaErrors(cudaFreeHost(h_2d_nx_nz_tmp3));

	checkCudaErrors(cudaFreeHost(h_2d_ny_nz_tmp0));
	checkCudaErrors(cudaFreeHost(h_2d_ny_nz_tmp1));
	checkCudaErrors(cudaFreeHost(h_2d_ny_nz_tmp2));
	checkCudaErrors(cudaFreeHost(h_2d_ny_nz_tmp3));

	for (int i = 0; i < 5; i++){
		checkCudaErrors(cudaStreamDestroy(stream[i]));	
	}
	//checkCudaErrors(cudaDeviceReset());
	return; 
}

void finalize_cuda_ipc(){
	/*
	if (n_east != -1){
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_ctsurf_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_ctbot_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_celg_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cjbar_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cadv_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cpvf_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cten_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp0_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp1_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp2_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp3_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp4_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp5_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp6_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp7_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp8_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp9_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp10_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp11_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp12_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp13_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_totx_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_toty_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_3d_tmp0_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_3d_tmp1_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_3d_tmp2_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_wubot_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_wvbot_east));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_aam_east));
	}
	if (n_west != -1){
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_ctsurf_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_ctbot_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_celg_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cjbar_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cadv_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cpvf_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_cten_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp0_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp1_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp2_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp3_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp4_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp5_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp6_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp7_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp8_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp9_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp10_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp11_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp12_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_2d_tmp13_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_totx_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_toty_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_3d_tmp0_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_3d_tmp1_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_3d_tmp2_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_wubot_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_wvbot_west));
		checkCudaErrors(cudaIpcCloseMemHandle((void*)d_aam_west));
	}
	*/
}

void init_device(){
#ifdef OPEN_MPI
	int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK")); 
	int local_size = atoi(getenv("OMPI_COMM_WORLD_LOCAL_SIZE")); 
	printf("local_rank = %d, local_size = %d\n\n",
			local_rank, local_size);
	sleep(10);
	//cudaSetDevice(local_rank%(local_size/2));
#endif
	
}

void init_device_impi(){
	const char *pciBus[4]={"0000:02:00.0", "0000:03:00.0",
						   "0000:83:00.0", "0000:84:00.0"};
	int deviceId;
	cudaDeviceGetByPCIBusId(&deviceId, (char*)pciBus[(my_task)%4]);
	
	cudaSetDevice(deviceId);
	printf("my_task %d setDevice %d\n", my_task, deviceId);
	sleep(10);
}


void end_device(){
	checkCudaErrors(cudaDeviceReset());
}



