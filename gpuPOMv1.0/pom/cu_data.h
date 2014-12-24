#ifndef CU_DATA_H
#define CU_DATA_H


#ifdef __NVCC__

#include<cuda_runtime.h>

/*
#define block_i 4
#define block_im1 (block_i-1)

#define block_j 8
#define block_jm1 (block_j-1)

#define block_k 31 
#define block_km1 (block_k-1)
*/
#define block_i_3D 32
#define block_j_3D 8
#define block_k_3D 1 


#define k_off (k*jm*im)
#define k_1_off ((k-1)*jm*im)
#define k_A1_off ((k+1)*jm*im)

#define kb_1_off ((kb-1)*jm*im)
#define kb_2_off ((kb-2)*jm*im)
#define kbm1_1_off ((kbm1-1)*jm*im)
#define kbm2_1_off ((kbm2-1)*jm*im)

#define ki_off (ki*jm*im)
#define ki_1_off ((ki-1)*jm*im)
#define ki_A1_off ((ki+1)*jm*im)

#define j_off (j*im)
#define j_A1_off ((j+1)*im)
#define j_1_off ((j-1)*im)

#define jm_1_off ((jm-1)*im)
#define jmm1_1_off ((jmm1-1)*im)

#define block_i_2D 32 
#define block_j_2D 4


#define checkCudaErrors(val)           check(val, __FILE__, __func__, __LINE__)
extern void check(cudaError_t err, const char* file, const char* func, unsigned line);

//1D const 
extern float *d_zz;//added by momentum3d
extern float *d_dz;//added by mode_interaction 
extern float *d_uabe, *d_uabw, //added by mode_external
			 *d_ele, *d_elw;
extern float *d_vabs, *d_vabn, 
			 *d_els, *d_eln;
extern float *d_dzz, *d_z;//added by mode_internal-->profq
//2D const
extern float *d_dx, *d_dy;//added by momentum3d 
extern float *d_aru, *d_arv;//added by momentum3d-->advct
extern float *d_aamfrz;//added by momentum3d 
extern float *d_art, *d_cor;//added by mode_external 
extern float *d_h, *d_fsm;//added by mode_external 
extern float *d_frz;//added by mode_internal-->bcond(4)


//3D const
extern float *d_tclim, *d_sclim;//added by mode_internal-->advt1


extern float *d_relax_aid;//added by mode_internal-->advt1

//2D
extern float *d_vfluxf, *d_e_atmos;//added by surface_forcing
extern float *d_swrad;//added by surface_forcing
extern float *d_dum, *d_dvm;//added by momentum3d 
extern float *d_dt, *d_d;//added by momentum3d 
extern float *d_adx2d, *d_ady2d;//added by mode_interaction 
extern float *d_drx2d, *d_dry2d;//added by mode_interaction 
extern float *d_aam2d;//added by mode_interaction 
extern float *d_ua, *d_va;//added by mode_interaction 
extern float *d_uab, *d_vab;//added by mode_interaction 
extern float *d_wubot, *d_wvbot;//added by mode_interaction 
extern float *d_advua, *d_advva;//added by mode_interaction 
extern float *d_el, *d_egf;//added by mode_interaction 
extern float *d_utf, *d_vtf;//added by mode_interaction 
extern float *d_cbc;//added by mode_interaction 

extern float *d_elb, *d_elf;//added by mode_external
extern float *d_wusurf, *d_wvsurf;//added by mode_external
extern float *d_etf;//added by mode_external
extern float *d_uaf, *d_vaf;//added by mode_external

extern float *d_utb, *d_vtb;//added by mode_internal
extern float *d_etb, *d_vfluxb;//added by mode_internal
extern float *d_tsurf;//added by mode_internal-->advt1
extern float *d_wtsurf, *d_wssurf;//added by mode_internal-->proft 
extern float *d_ssurf;//added by mode_internal-->proft

extern float *d_tbe, *d_sbe, *d_tbw, *d_sbw;//added by mode_internal-->bcond(4)
extern float *d_tbs, *d_sbs, *d_tbn, *d_sbn;//added by mode_internal-->bcond(4)

extern float *d_egb;//added by mode_internal-->advu

extern float *d_et;//added by mode_internal-->kernel_4

extern float *d_uab_mean, *d_vab_mean;//added by store_mean
extern float *d_elb_mean;//added by store_mean
extern float *d_wusurf_mean, *d_wvsurf_mean;//added by store_mean
extern float *d_wtsurf_mean, *d_wssurf_mean;//added by store_mean


extern float *d_usrf_mean, *d_vsrf_mean;//add by store_surf_mean
extern float *d_elsrf_mean;//add by store_surf_mean 
extern float *d_uwsrf_mean, *d_vwsrf_mean;//add by store_surf_mean
extern float *d_uwsrf, *d_vwsrf;//add by store_surf_mean
extern float *d_utf_mean, *d_vtf_mean;//add by store_surf_mean
extern float *d_celg_mean, *d_ctsurf_mean;//add by store_surf_mean
extern float *d_celg, *d_ctsurf;//add by store_surf_mean
extern float *d_cpvf_mean, *d_cjbar_mean;//add by store_surf_mean
extern float *d_cpvf, *d_cjbar;//add by store_surf_mean
extern float *d_cadv_mean, *d_cten_mean;//add by store_surf_mean
extern float *d_cadv, *d_cten;//add by store_surf_mean
extern float *d_ctbot_mean;//add by store_surf_mean
extern float *d_ctbot;//add by store_surf_mean

extern float *d_ctot, *d_totx, *d_toty;//add by mode_external-->vort


//3D
extern float *d_w;//added by surface_forcing
extern float *d_u, *d_v;//added by momentum3d 
extern float *d_ub, *d_vb;//added by momentum3d 
extern float *d_aam, *d_rho, *d_rmean;//added by momentum3d 
extern float *d_advx, *d_advy;//added by momentum3d 
extern float *d_drhox, *d_drhoy;//added by momentum3d 

extern float *d_q2b, *d_q2;//added by mode_internal 
extern float *d_q2lb, *d_q2l;//added by mode_internal 
extern float *d_uf, *d_vf;//added by mode_internal 
extern float *d_kq, *d_l;//added by mode_internal-->profq 
//d_l is actually a tmp array

extern float *d_t, *d_s;//added by mode_internal-->profq 
extern float *d_km, *d_kh;//added by mode_internal-->profq 
extern float *d_tb, *d_sb; //added by mode_internal-->advt1

extern float *d_tobw, *d_sobw, //added by mode_internal-->bcond(4)
			 *d_tobe, *d_sobe,
			 *d_tobs, *d_sobs,
			 *d_tobn, *d_sobn;

extern float *d_u_mean, *d_v_mean, *d_w_mean;//added by store_mean
extern float *d_t_mean, *d_s_mean;//added by store_mean
extern float *d_rho_mean, *d_kh_mean, *d_km_mean;//added by store_mean
extern float *d_ustks, *d_vstks;//added by store_mean

extern float *d_xstks_mean, *d_ystks_mean;//add by store_surf_mean
extern float *d_xstks, *d_ystks;//add by store_surf_mean



extern float *d_3d_tmp0, *d_3d_tmp1, 
			 *d_3d_tmp2, *d_3d_tmp3;

extern float *d_3d_tmp4, *d_3d_tmp5, 
			 *d_3d_tmp6, *d_3d_tmp7,
			 *d_3d_tmp8, *d_3d_tmp9,
			 *d_3d_tmp10, *d_3d_tmp11;

extern float *d_2d_tmp0, *d_2d_tmp1;
extern float *d_2d_tmp2, *d_2d_tmp3,
			 *d_2d_tmp4, *d_2d_tmp5,
			 *d_2d_tmp6, *d_2d_tmp7,
			 *d_2d_tmp8, *d_2d_tmp9,
			 *d_2d_tmp10, *d_2d_tmp11,
			 *d_2d_tmp12, *d_2d_tmp13;

extern float *d_1d_ny_tmp0, *d_1d_ny_tmp1,
			 *d_1d_ny_tmp2, *d_1d_ny_tmp3;

//////////////////////////////////////////////
//data copy back to host and then mpi
extern float *h_1d_nx_tmp0, *h_1d_nx_tmp1,
			 *h_1d_nx_tmp2, *h_1d_nx_tmp3;
extern float *h_1d_ny_tmp0, *h_1d_ny_tmp1,
			 *h_1d_ny_tmp2, *h_1d_ny_tmp3;
//////////////////////////////////////////////


extern float *d_2d_ny_nz_tmp0, *d_2d_ny_nz_tmp1,
			 *d_2d_ny_nz_tmp2, *d_2d_ny_nz_tmp3;

extern float *d_2d_nx_nz_tmp0, *d_2d_nx_nz_tmp1,
			 *d_2d_nx_nz_tmp2, *d_2d_nx_nz_tmp3;

//////////////////////////////////////////////
//data copy back to host and then mpi
extern float *h_2d_ny_nz_tmp0, *h_2d_ny_nz_tmp1,
			 *h_2d_ny_nz_tmp2, *h_2d_ny_nz_tmp3;

extern float *h_2d_nx_nz_tmp0, *h_2d_nx_nz_tmp1,
			 *h_2d_nx_nz_tmp2, *h_2d_nx_nz_tmp3;
//////////////////////////////////////////////

extern float *d_ctsurf_east, *d_ctsurf_west,
			 *d_ctbot_east, *d_ctbot_west,
			 *d_celg_east, *d_celg_west,
			 *d_cjbar_east, *d_cjbar_west,
			 *d_cadv_east, *d_cadv_west,
			 *d_cpvf_east, *d_cpvf_west,
			 *d_cten_east, *d_cten_west;

////////////////////////////////////////////////////
extern float *d_2d_tmp0_east, *d_2d_tmp0_west,
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

extern float *d_2d_tmp0_south, *d_2d_tmp0_north,
			 *d_2d_tmp1_south, *d_2d_tmp1_north;
////////////////////////////////////////////////////

extern float *d_totx_east, *d_totx_west,
			 *d_toty_east, *d_toty_west;

////////////////////////////////////////////////////
extern float *d_3d_tmp0_east, *d_3d_tmp0_west,
			 *d_3d_tmp1_east, *d_3d_tmp1_west,
			 *d_3d_tmp2_east, *d_3d_tmp2_west;
extern float *d_3d_tmp0_south, *d_3d_tmp0_north,
			 *d_3d_tmp1_south, *d_3d_tmp1_north,
		     *d_3d_tmp2_south, *d_3d_tmp2_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_wubot_east, *d_wubot_west,
			 *d_wubot_east_most, *d_wubot_west_most,
			 *d_wvbot_east, *d_wvbot_west,
			 *d_wvbot_east_most, *d_wvbot_west_most;

extern float *d_wubot_south, *d_wubot_north,
		     *d_wvbot_south, *d_wvbot_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_aam_east, *d_aam_west;
extern float *d_aam_south, *d_aam_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_elf_east, *d_elf_west;
extern float *d_elf_east_most, *d_elf_west_most;
extern float *d_elf_south, *d_elf_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_uaf_east, *d_uaf_west;
extern float *d_uaf_east_most, *d_uaf_west_most;
extern float *d_uaf_south, *d_uaf_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_vaf_east, *d_vaf_west;
extern float *d_vaf_east_most, *d_vaf_west_most;
extern float *d_vaf_south, *d_vaf_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_w_east, *d_w_west;
extern float *d_w_east_most, *d_w_west_most;
extern float *d_w_south, *d_w_north;
////////////////////////////////////////////////////


////////////////////////////////////////////////////
extern float *d_uf_east, *d_uf_west,
			 *d_uf_south, *d_uf_north,
			 *d_vf_east, *d_vf_west,
			 *d_vf_south, *d_vf_north,
	  		 *d_uf_east_most, *d_uf_west_most,
	  		 *d_vf_east_most, *d_vf_west_most,
	  		 *d_kh_east_most, *d_kh_west_most,
	  		 *d_km_east_most, *d_km_west_most,
	  		 *d_kq_east_most, *d_kq_west_most;
////////////////////////////////////////////////////


//////////////////////////////////////////
//////////////////////////////////////////

extern cudaStream_t stream[5];

extern dim3 threadPerBlock;
extern dim3 blockPerGrid;

extern dim3 threadPerBlock_inner;
extern dim3 blockPerGrid_inner;

extern dim3 threadPerBlock_ew_32;
extern dim3 blockPerGrid_ew_32;

extern dim3 threadPerBlock_sn_32;
extern dim3 blockPerGrid_sn_32;

extern dim3 threadPerBlock_ew_b1;
extern dim3 blockPerGrid_ew_b1;

extern dim3 threadPerBlock_ew_b2;
extern dim3 blockPerGrid_ew_b2;

extern dim3 threadPerBlock_sn_b1;
extern dim3 blockPerGrid_sn_b1;

extern dim3 threadPerBlock_sn_b2;
extern dim3 blockPerGrid_sn_b2;

//persistent variables
//scalar const
/*
extern int *d_n_east, *d_n_west, *d_n_north, *d_n_south;
extern int *d_my_task;
extern int *d_kb, *d_jm, *d_im;
extern int *d_nitera, *d_mode, *d_ntp;

extern float *d_sw, *d_dti2, *d_tprni, *d_grav, 
			 *d_tbias, *d_sbias, *d_rhoref, *d_umol,
			 *d_kappa, *d_small;
extern float *d_dti;
*/

////1D const
//extern float *d_dzz, *d_dz, *d_zz, *d_z;
//extern int *d_j_global;
//extern float *d_uabe, *d_uabw, *d_ele, *d_elw;
//extern float *d_vabs, *d_vabn, *d_els, *d_eln;
//
//extern float *d_aam_aid;
//
////2D const
//extern float *d_fsm, *d_aru, *d_arv, *d_art;
//extern float *d_dx, *d_dy, *d_dum, *d_dvm;
//extern float *d_cor, *d_cbc,  *d_h;
//extern float *d_tsurf, *d_ssurf;
//extern float *d_tbe, *d_sbe, *d_tbw, *d_sbw;
//extern float *d_tbs, *d_sbs, *d_tbn, *d_sbn;
////3D const
//extern float *d_rmean, *d_tclim, *d_sclim;
//////////////////////////////////////////////////
////kq is a special variable, 
////because it is only referenced and modified in profq
////but we have to make it a global variable 
////for the value is useful in the next iteration
//////////////////////////////////////////////////
//extern float *d_kq;//added by profq
//
//
//
///////////////////////////////////////////////////////
//
//extern float *d_dt;
//extern float *d_etf, *d_aam;
//
//extern float *d_u, *d_v, *d_etb, *d_w;//added by advt2
//extern float *d_advua, *d_advva, *d_d, *d_ua, *d_va, 
//			 *d_fluxua, *d_fluxva, *d_uab, *d_vab, *d_aam2d;//added by advave
//extern float *d_advx, *d_advy, *d_ub, *d_vb; //added by advct
//extern float *d_egf, *d_egb, *d_e_atmos, *d_drhox, *d_uf;//added by advu
//extern float *d_drhoy, *d_vf;//added by advv
//extern float *d_rho;//added by baropg 
//extern float *d_vfluxf, *d_vfluxb;//added by vertvl
//extern float *d_wusurf, *d_wvsurf, *d_wubot, *d_wvbot;//added by profq
//extern float *d_t, *d_s, *d_q2b, *d_q2lb, *d_l, *d_km, *d_kh, *d_q2;//added by profq
//extern float *d_swrad;//added by proft
//extern float *d_wtsurf, *d_wssurf;//added by surface_forcing 
//extern float *d_el;//added by mode_interaction
//extern float *d_elb;//added by mode_external
//extern float *d_utb, *d_vtb, *d_q2l;//added by mode_internal
//extern float *d_tb, *d_sb, *d_et; //added by mode_internal
//
//
//		
//
///////////////////////////////////////////////////////
////variable
////need not copy-in
//extern float *d_adx2d, *d_ady2d, *d_drx2d, *d_dry2d;// *d_aam2d;
//extern float *d_elf, *d_utf, *d_vtf;//added by mode_external
//extern float *d_uaf, *d_vaf; //added by mode_external
//
//
//
//
///////////////////////////////////////////////////////
////local variables
//extern float *d_3d_tmp0, *d_3d_tmp1, *d_3d_tmp2, 
//			 *d_3d_tmp3, *d_3d_tmp4, *d_3d_tmp5,
//			 *d_3d_tmp6; //*d_3d_tmp7, *d_3d_tmp8,
//			 //*d_3d_tmp9, *d_3d_tmp10, *d_3d_tmp11,
//			 //*d_3d_tmp12;
//
///*
//extern double *d_3d_tmp0_d, *d_3d_tmp1_d, 
//			  *d_3d_tmp2_d, *d_3d_tmp3_d;
//*/
//
//extern float *d_2d_tmp0, *d_2d_tmp1, *d_2d_tmp2; 

/////////////////////////////////////////////////////
//advt2 local variable
//extern float *d_xflux_advt2, *d_yflux_advt2, *d_zflux_advt2;
//extern float *d_fbmem_advt2, *d_eta_advt2;
//extern float *d_xmassflux_advt2, *d_ymassflux_advt2, *d_zwflux_advt2;
//extern float *d_f, *d_ff, *d_fb, *d_fclim;//these are really ugly, for they are formal parameter
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//advave local variable
//extern float *d_tps_advave, *d_curv2d_advave;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//advct local variable
//extern float *d_xflux_advct, *d_yflux_advct, *d_curv_advct;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//advq local variable
//extern float *d_xflux_advq, *d_yflux_advq;
//extern float *d_q_advq, *d_qb_advq, *d_qf_advq;//these are really ugly, for they are formal parameter
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//advt1 local variable
//extern float *d_xflux_advt1, *d_yflux_advt1, *d_zflux_advt1;
//extern float *d_ff_advt1, *d_f_advt1, *d_fb_advt1, *d_fclim_advt1;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//dens local variable
//extern float *d_si_dens, *d_ti_dens, *d_rhoo_dens;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//vertvl local variable
//extern float *d_xflux_vertvl, *d_yflux_vertvl;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//profq local variable
//extern float *d_utau2_profq, *d_l0_profq, *d_dh_profq;
//extern float *d_a_profq, *d_c_profq, *d_ee_profq, *d_gg_profq;
//extern float *d_boygr_profq, *d_prod_profq, *d_const1_profq;
//extern float *d_cc_profq, *d_gh_profq, *d_stf_profq, *d_dtef_profq; 
//extern float *d_ghc_profq, *d_sh_profq, *d_sm_profq; 
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//proft local variable
//extern float *d_f_proft, *d_wfsurf_proft, *d_fsurf_proft;
//extern double *d_a_proft, *d_c_proft, *d_ee_proft, *d_gg_proft;
//extern float *d_dh_proft, *d_rad_proft;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//profu local variable
//extern double *d_a_profu, *d_c_profu, *d_ee_profu, *d_gg_profu;
//extern float *d_dh_profu, *d_tps_profu;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//profv local variable
//extern double *d_a_profv, *d_c_profv, *d_ee_profv, *d_gg_profv;
//extern float *d_dh_profv, *d_tps_profv;
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//mode_internal local variable
//extern float *d_tps_mode_internal;

#endif

#ifdef __NVCC__
extern "C" {
#endif

	void init_cuda_gpu(float* fsm,   float* aru, float* arv, 
					   float *dzz,   float* h,   float *dx,
				       float *dy,    float *dum,
					   float *dvm,   float *art,
					   float *dz,    float *cor,
					   float *rmean, float *zz, 
					   float *z,     float *cbc, float *kq);

	void init_cuda_scalar_const(int n_east, int n_west, int n_north, int n_south,
								 int my_task, int kb, int jm, int im,
								 int nitera, int mode, int ntp,
								 float sw, float dit2, float tprni, float grav, 
								 float tbias, float sbias, float rhoref, float umol,
								 float kappa, float small);
	
	/*
	void init_cuda_1d_const(float *z,  float *zz,
							 float *dz, float *dzz,
							 int *j_global,
							 float *uabe, float *uabw, 
							 float *ele, float *elw,
							 float *vabs, float *vabn,
							 float *els, float *eln);
	*/
	void init_cuda_1d_const();

	/*
	void init_cuda_2d_const(float* aru, float* arv, float *art,
							 float *dx,  float *dy,
							 float *dum, float *dvm, 
							 float* h,   float *cor, 
							 float* fsm, float *cbc,
							 float *tsurf, float *ssurf,
							 float *tbe, float *sbe,
							 float *tbw, float *sbw,
							 float *tbs, float *sbs,
							 float *tbn, float *sbn);
	*/
	void init_cuda_2d_const();

	void init_cuda_3d_const();

	void init_cuda_1d_var();


	//void init_cuda_2d_var(float *vfluxf);

	void init_cuda_2d_var();

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
					   float *egb, float *et);
	*/


	void init_cuda_3d_var();
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
					   float *km, float *kh);
	*/

	void init_cuda_local();

	void init_device();

	void init_device_impi();

	//void init_cuda_ipc();
	void exchangeMemHandle();
	void openMemHandle();

	void init_cuda_peer();

	void init_cuda_pinned_memory();

	void finalize_cuda_gpu();

	void finalize_cuda_ipc();

	void end_device();

	//void get_time_gpu();

	/*
	void surface_forcing_gpu();

	void lateral_viscosity_gpu();
		
	void surfacing_forcing_gpu();

	void mode_interaction_gpu();

	void mode_external_gpu();

	void mode_internal_gpu();

	void bcond_gpu(int idx);
	*/
#ifdef __NVCC__
}
#endif



#endif
