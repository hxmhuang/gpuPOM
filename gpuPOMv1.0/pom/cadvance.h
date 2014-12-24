#ifndef CADVANCE_H
#define CADVANCE_H


//void advance_main();

//void advance_c_(int *f_iint);
void advance();

void get_time();

/*
void get_time_(float *f_time, 
               float *f_ramp,
			   int *f_iint);
*/
	
void surface_forcing();

/*
void surface_forcing_(float f_vfluxf[][i_size], 
					  float f_t[][j_size][i_size],
					  float f_e_atmos[][i_size],
					  float f_w[][j_size][i_size],
					  float f_swrad[][i_size]);
*/
	
/*
void momentum3d_(float f_advx[][j_size][i_size],
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
				float f_d[][i_size]);
*/
void momentum3d();

//void lateral_viscosity();

/*
void mode_interaction_(float f_adx2d[][i_size], float f_ady2d[][i_size],
					  float f_drx2d[][i_size], float f_dry2d[][i_size],
					  float f_aam2d[][i_size], float f_egf[][i_size],
					  float f_utf[][i_size], float f_vtf[][i_size],
					  float f_advua[][i_size], float f_advva[][i_size],
					  float f_wubot[][i_size], float f_wvbot[][i_size],

					  float f_advx[][j_size][i_size], float f_advy[][j_size][i_size],
					  float f_drhox[][j_size][i_size], float f_drhoy[][j_size][i_size],
					  float f_aam[][j_size][i_size], float f_d[][i_size],
					  float f_ua[][i_size], float f_va[][i_size],
					  float f_uab[][i_size], float f_vab[][i_size],
					  float f_el[][i_size], float f_aamfrz[][i_size]);
*/

void mode_interaction();
	
void mode_external();
	
void mode_internal();

void print_section();

void check_velocity();

void store_mean();

void store_surf_mean();

#endif
