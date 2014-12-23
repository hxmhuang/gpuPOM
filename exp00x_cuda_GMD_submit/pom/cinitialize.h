#ifndef c_INITIALIZE_H
#define C_INITIALIZE_H

#include"data.h"

void initialize();

void read_input();

void initialize_arrays();

void read_grid();
/*
void read_grid_(float f_cor[][i_size], float f_art[][i_size],
				float f_aru[][i_size], float f_arv[][i_size],
				float f_d[][i_size], float f_dt[][i_size],
				int *f_nsw, int *f_nse,
				int *f_nss, int *f_nsn,
				float f_east_e[][i_size],
				float f_north_e[][i_size],
				float f_rot[][i_size],
				float f_dx[][i_size],
				float f_dy[][i_size],
				float f_h[][i_size],
				float f_fsm[][i_size],
				float f_dum[][i_size],
				float f_dvm[][i_size],
				float *f_alonc, float *f_alatc,
		        float *f_z, float *f_zz, 
				float *f_dz, float *f_dzz,
				float f_east_c[][i_size], 
				float f_north_c[][i_size],
				float f_east_u[][i_size], 
				float f_north_u[][i_size],
				float f_east_v[][i_size], 
				float f_north_v[][i_size]);
*/

void specify_grid();
/*
void specify_grid_(int *f_nsw, int *f_nse,
				  int *f_nss, int *f_nsn,
				  float f_east_e[][i_size],
				  float f_north_e[][i_size],
				  float f_rot[][i_size],
				  float f_dx[][i_size],
				  float f_dy[][i_size],
				  float f_h[][i_size],
				  float f_fsm[][i_size],
				  float f_dum[][i_size],
				  float f_dvm[][i_size],
				  float *f_alonc, float *f_alatc,
				  float *f_z, float *f_zz,
				  float f_east_c[][i_size],
				  float f_north_c[][i_size],
				  float f_east_u[][i_size],
				  float f_north_u[][i_size],
				  float f_east_v[][i_size],
				  float f_north_v[][i_size]);
*/

void east_north_ecuv();

void depth(float *z, float *zz,
		   int kb, int logornolog);

void initial_conditions();
/*
void initial_conditions_(float f_tb[][j_size][i_size], 
						 float f_t[][j_size][i_size],
						 float f_sb[][j_size][i_size],
						 float f_s[][j_size][i_size],
						 float f_rho[][j_size][i_size],
						 float f_rmean[][j_size][i_size],
						 float f_tclim[][j_size][i_size],
						 float f_sclim[][j_size][i_size],
						 float f_tobw[][j_size][nfw],
						 float f_sobw[][j_size][nfw],
						 float f_tobe[][j_size][nfe],
						 float f_sobe[][j_size][nfe],
						 float f_tobs[][nfs][i_size],
						 float f_sobs[][nfs][i_size],
						 float f_tobn[][nfn][i_size],
						 float f_sobn[][nfn][i_size],
						 float f_tbe[][j_size],
						 float f_tbw[][j_size],
						 float f_sbe[][j_size],
						 float f_sbw[][j_size],
						 float f_tbn[][i_size],
						 float f_tbs[][i_size],
						 float f_sbn[][i_size],
						 float f_sbs[][i_size]);
*/

/*
void lateral_boundary_conditions_(float *f_rfe, float *f_rfw,
							     float *f_rfn, float *f_rfs,
								 float f_cor[][i_size]);
*/
void lateral_boundary_conditions();

void read_tide();

int judge_inout(int i_in, int j_in,
				int imin_in, int imax_in,
				int jmin_in, int jmax_in);

void read_trajdrf();
//void read_trajdrf_();

void bfrz(int mw, int me, int ms, int mn,
		  int nw, int ne, int ns, int nn,
		  int im, int jm, int nu, float frz[][i_size]);

/*
void bfrz_(int *f_mw, int *f_me, int *f_ms, int *f_mn,
		  int *f_nw, int *f_ne, int *f_ns, int *f_nn,
		  int *f_im, int *f_jm, int *f_nu, float f_frz[][i_size],
		  float *f_rdisp);
*/

void update_initial();

/*
void update_initial_(float f_ua[][i_size], float f_va[][i_size],
					 float f_el[][i_size], float f_et[][i_size],
					 float f_etf[][i_size], float f_d[][i_size],
					 float f_dt[][i_size], float f_w[][j_size][i_size],
					 float f_drx2d[][i_size], float f_dry2d[][i_size],
					 float f_l[][j_size][i_size],
					 float f_q2b[][j_size][i_size],
					 float f_q2lb[][j_size][i_size],
					 float f_kh[][j_size][i_size],
					 float f_km[][j_size][i_size],
					 float f_kq[][j_size][i_size],
					 float f_aam[][j_size][i_size],
					 float f_drhox[][j_size][i_size],
					 float f_drhoy[][j_size][i_size],
					 float f_q2[][j_size][i_size],
					 float f_q2l[][j_size][i_size],
					 float f_t[][j_size][i_size],
					 float f_s[][j_size][i_size],
					 float f_u[][j_size][i_size],
					 float f_v[][j_size][i_size]);
*/

void bottom_friction();
/*
void bottom_friction_(float f_cbc[][i_size],
					  float f_aamfrz[][i_size]);
*/

/*
void incmix_(float aam[][j_size][i_size], 
			 int *f_im, int *f_jm, int *f_kb,
			 float x[][i_size], float y[][i_size]);
*/

void incmix(float aam[][j_size][i_size], 
			int im, int jm, int kb,
			float x[][i_size], float y[][i_size]);

void ztosig(float *zs, float tb[][j_size][i_size], float *zz,
			float h[][i_size], float t[][j_size][i_size],
			int ks);

void splinc(float *x, float *y, int n, float yp1, float ypn,
			 float *xnew, float *ynew, int m);

void splint(float *xa, float *ya, float *y2a, 
		    int n, float x, float *y);


#endif
