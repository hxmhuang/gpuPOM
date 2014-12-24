#ifndef CSOLVER_GPU_H
#define CSOLVER_GPU_H

#ifdef __NVCC__
extern "C" {
#endif

	void initial_constant_csolver(float *c_dzz, float *c_dz,
								  float *c_zz, float *z);

	/*
	void smol_adif_gpu(float xmassflux[][j_size][i_size], 
					   float ymassflux[][j_size][i_size], 
					   float zwflux[][j_size][i_size], 
					   float ff[][j_size][i_size]);
	*/

	void smol_adif_gpu(float *d_xmassflux, 
				       float *d_ymassflux, 
				       float *d_zwflux, 
				       float *d_ff, 
				       float *d_dt);
	/*
	void advt2_gpu(float fb[][j_size][i_size],   float f[][j_size][i_size],
				   float fclim[][j_size][i_size],float ff[][j_size][i_size],
			       float etb[][i_size],          float u[][j_size][i_size],
			       float v[][j_size][i_size],    float etf[][i_size], 
			       float aam[][j_size][i_size],  float w[][j_size][i_size],
			       float dt[][i_size]);
	*/
	/*
	void advt2_gpu(float *d_fb,   float *d_f,
				   float *d_fclim,float *d_ff,
			       float *d_u, float *d_v, float *d_w,  
			       float *d_etb, float *d_etf, 
			       float *d_aam, float *d_dt);
	void advt2_gpu(float *d_fb,   float *d_f,
				   float *d_fclim,float *d_ff);
	void advt2_gpu(float *d_fb,   float *d_f,
				   float *d_fclim,float *d_ff,
			       char var);
	*/

	void advt2_gpu(float *d_fb,   float *d_f,
				   float *d_fclim,float *d_ff,
				   float *d_ff_east, float *d_ff_west,
				   float *d_ff_south, float *d_ff_north,
				   char var);

	/*
	void advave_gpu(float advua[][i_size], float d[][i_size],
					float ua[][i_size], float va[][i_size],
					float fluxua[][i_size], float fluxva[][i_size],
					float uab[][i_size], float aam2d[][i_size],
					float vab[][i_size], float advva[][i_size],
					float wubot[][i_size], float wvbot[][i_size]);
	*/

	/*
	void advave_gpu(float *d_advua, float *d_advva,
					float *d_fluxua, float *d_fluxva,
					float *d_wubot, float *d_wvbot,
					float *d_d, float *d_aam2d,
					float *d_ua, float *d_va,
					float *d_uab, float *d_vab);
	*/
	void advave_gpu();

	/*
	void advct_gpu(float advx[][j_size][i_size], float v[][j_size][i_size],
				   float u[][j_size][i_size], float dt[][i_size], 
				   float ub[][j_size][i_size], float aam[][j_size][i_size],
				   float vb[][j_size][i_size], float advy[][j_size][i_size]);
	*/

	/*
	void advct_gpu(float *d_advx,
				   float *d_advy,
			       float *d_u, float *d_ub, 
				   float *d_v, float *d_vb, 
				   float *d_dt, float *d_aam);
	*/
	void advct_gpu();

	/*
	void advq_gpu(float qb[][j_size][i_size], float q[][j_size][i_size],
				  float qf[][j_size][i_size], float u[][j_size][i_size],
				  float dt[][i_size], float v[][j_size][i_size],
				  float aam[][j_size][i_size], float w[][j_size][i_size],
				  float etb[][i_size], float etf[][i_size]);
	*/
	
	/*
	void advq_gpu(float *d_qb, float *d_q, float *d_qf, 
				  float *d_u, float *d_v, float *d_w,
				  float *d_etb, float *d_etf,
				  float *d_aam, float *d_dt);
	*/
	void advq_gpu(float *d_qb, float *d_q, float *d_qf); 

	/*
	void advt1_gpu(float fb[][j_size][i_size], float f[][j_size][i_size],
				   float fclim[][j_size][i_size], float ff[][j_size][i_size],
				   float dt[][i_size], float u[][j_size][i_size],
			       float v[][j_size][i_size], float aam[][j_size][i_size],
			       float w[][j_size][i_size], float etb[][i_size],
			       float etf[][i_size]);
	*/
	
	/*
	void advt1_gpu(float *d_fb, float *d_f,
				   float *d_fclim, float *d_ff,
				   float *d_u, float *d_v, float *d_w, 
			       float *d_etb, float *d_etf, 
			       float *d_aam, float *d_dt);
	*/
	void advt1_gpu(float *d_fb, float *d_f,
				   float *d_fclim, float *d_ff,
				   char var);

	/*
	void advu_gpu(float uf[][j_size][i_size], float w[][j_size][i_size],
				  float u[][j_size][i_size], float advx[][j_size][i_size],
				  float dt[][i_size], float v[][j_size][i_size],
				  float egf[][i_size], float egb[][i_size],
				  float e_atmos[][i_size], float drhox[][j_size][i_size],
				  float etb[][i_size], float ub[][j_size][i_size],
				  float etf[][i_size]);
	*/

	/*
	void advu_gpu(float *d_ub, float *d_u, float *d_uf, 
				  float *d_v, float *d_w, 
				  float *d_egb, float *d_egf, 
				  float *d_etb, float *d_etf, 
				  float *d_advx, float *d_drhox,
				  float *d_e_atmos, float *d_dt);
	*/
	void advu_gpu();

	/*
	void advv_gpu(float vf[][j_size][i_size], float w[][j_size][i_size],
				  float u[][j_size][i_size], float advy[][j_size][i_size],
				  float dt[][i_size], float v[][j_size][i_size],
				  float egf[][i_size], float egb[][i_size],
				  float e_atmos[][i_size], float drhoy[][j_size][i_size],
				  float etb[][i_size], float vb[][j_size][i_size],
				  float etf[][i_size]);
	*/

	/*
	void advv_gpu(float *d_vb, float *d_v, float *d_vf, 
				  float *d_u, float *d_w,
				  float *d_egb, float *d_egf, 
				  float *d_etb, float *d_etf,
				  float *d_advy, float *d_drhoy,
				  float *d_e_atmos, float *d_dt);
	*/
	void advv_gpu();


	/*
	void baropg_gpu(float rho[][j_size][i_size], 
					float drhox[][j_size][i_size],
					float dt[][i_size], 
					float drhoy[][j_size][i_size], 
					float ramp);
	*/

	/*
	void baropg_gpu(float *d_drhox, float *d_drhoy,
					float *d_dt, float *d_rho, 
					float ramp);
	*/
	void baropg_gpu();

	/*
	void dens_gpu(float si[][j_size][i_size], 
			      float ti[][j_size][i_size],
				  float rhoo[][j_size][i_size]);
	*/

	void dens_gpu(float *d_si, float *d_ti, float *d_rhoo);

	/*
	void profq_gpu(float etf[][i_size], float wusurf[][i_size],
				   float wvsurf[][i_size],float wubot[][i_size],
				   float wvbot[][i_size], float q2b[][j_size][i_size],
				   float q2lb[][j_size][i_size], float u[][j_size][i_size],
				   float v[][j_size][i_size], float km[][j_size][i_size], 
				   float uf[][j_size][i_size], float vf[][j_size][i_size],
				   float q2[][j_size][i_size], float dt[][i_size],
				   float kh[][j_size][i_size], float t[][j_size][i_size],
				   float s[][j_size][i_size], float rho[][j_size][i_size]);
	*/

	/*
	void profq_gpu(float *d_uf, float *d_vf,
				   float *d_km, float *d_kh, 
				   float *d_q2b, float *d_q2lb, 
				   float *d_wusurf, float *d_wvsurf,
				   float *d_wubot, float *d_wvbot, 
				   float *d_u, float *d_v, 
				   float *d_t, float *d_s,
				   float *d_dt, float *d_rho,
				   float *d_q2, float *d_etf);
	*/
	void profq_gpu();

	/*
	void proft_gpu(float f[][j_size][i_size], float wfsurf[][i_size],
				   float fsurf[][i_size], int nbc,
			       float etf[][i_size], float kh[][j_size][i_size],
				   float swrad[][i_size]);
	*/
	
	/*
	void proft_gpu(float *d_f, float *d_wfsurf,
				   float *d_fsurf, 
			       float *d_etf, float *d_kh,
				   float *d_swrad, int nbc);
	*/
	void proft_gpu(float *d_f, float *d_wfsurf,
				   float *d_fsurf, int nbc);

	/*
	void profu_gpu(float etf[][i_size],float km[][j_size][i_size],
				   float wusurf[][i_size],float uf[][j_size][i_size],
				   float vb[][j_size][i_size],float ub[][j_size][i_size],
				   float wubot[][i_size]);
	*/

	/*
	void profu_gpu(float *d_ub, float *d_uf, 
				   float *d_wusurf, float *d_wubot,
				   float *d_vb, float *d_km,
				   float *d_etf);
	*/
	void profu_gpu();

	/*
	void profv_gpu(float etf[][i_size],float km[][j_size][i_size],
				   float wvsurf[][i_size],float vf[][j_size][i_size],
				   float ub[][j_size][i_size],float vb[][j_size][i_size],
				   float wvbot[][i_size]);
	*/

	/*
	void profv_gpu(float *d_vb, float *d_vf, 
				   float *d_wvsurf, float *d_wvbot,
				   float *d_ub, float *d_km,
				   float *d_etf);
	*/
	void profv_gpu();

	/*
	void vertvl_gpu(float dt[][i_size], float u[][j_size][i_size],
					float v[][j_size][i_size],float vfluxb[][i_size],
					float vfluxf[][i_size], float w[][j_size][i_size],
					float etf[][i_size],float etb[][i_size]);
	*/

	/*
	void vertvl_gpu(float *d_u, float *d_v, float *d_w,
					float *d_vfluxb, float *d_vfluxf, 
					float *d_etf,float *d_etb,
					float *d_dt);
	*/
	void vertvl_gpu();

	void vertvl_overlap_bcond(cudaStream_t &stream_inner,
							  cudaStream_t &stream_ew,
							  cudaStream_t &stream_sn);

	void vort_gpu();

	void advq_fusion_gpu(float *d_qub, float *d_qu, float *d_quf, 
						 float *d_qvb, float *d_qv, float *d_qvf);

	void proft_fusion_gpu(
					float *d_f_u, float *d_wfsurf_u,
					float *d_fsurf_u, int nbc_u,
					float *d_f_v, float *d_wfsurf_v,
					float *d_fsurf_v, int nbc_v);

	void proft_fusion_overlap_bcond(
					float *d_f_u, float *d_wfsurf_u,
					float *d_fsurf_u, int nbc_u,
					float *d_f_v, float *d_wfsurf_v,
					float *d_fsurf_v, int nbc_v,
					cudaStream_t &stream_inner,
					cudaStream_t &stream_ew,
					cudaStream_t &stream_sn);

	void profuv_fusion_gpu();

	void profuv_fusion_overlap_bcond();

	void profq_overlap_bcond();

	void advuv_fusion_gpu();

#ifdef __NVCC__
}
#endif

#endif
