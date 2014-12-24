#ifndef DATA_H
#define DATA_H

#include<time.h>
#include"mpi.h"
#include"utils.h"

#define ABS(i) (i>=0? i:0-i)
#define MAX(i,j) ((i)>=(j)? (i):(j))
#define MIN(i,j) (i<=j? i:j)
#define NINT(i) ((int)(i+0.5f))



/*
 * below are const variables during advance steps
 */

extern MPI_Comm pom_comm, pom_comm_coarse;
extern int origin_task;
extern int my_task, 
		   master_task, error_status, n_proc,
		   im_global, im_local, jm_global, jm_local,
		   im, imm1, imm2, jm, jmm1, jmm2, kb, kbm1, kbm2,
		   n_east, n_west, n_north, n_south;

extern int im_coarse, im_local_coarse, im_global_coarse;
extern int jm_coarse, jm_local_coarse, jm_global_coarse;
extern int x_division, y_division;
extern int iprints;
extern struct tm dtime0, dtime;

extern int mode, nadv, nitera, npg, dte, isplit, nread_rst,
		   iperx, ipery, n1d, ngrid;

extern int nse, nsw, nsn, nss; 

extern int lramp, ntp, nbct, nbcs, lspadv, iend, iprint, irestart,
	       ispadv, iint;

extern int nb, np;

extern int calc_wind, calc_tsforce, calc_river, calc_assim,
	       calc_assimdrf, calc_tsurf_mc, calc_tide, calc_trajdrf,
		   tracer_flag, calc_stokes, calc_vort, output_flag,
		   SURF_flag,
		   calc_interp; //!fhx:interp_flag

extern int iout, iouts, ioutv;

extern float rhoref, tbias, sbias, grav, kappa, z0b,
			 cbcmin, cbcmax, horcon, tprni, umol, vmaxl,
			 slmax, smoth, alpha, aam_init, pi, dti, dti2, dte2,
			 isp2i, small, ispi, sw;

extern float rfw, rfe, rfs, rfn; 

extern float days, prtd1, prtd2, write_rst;

extern float lono, lato, xs, ys, fak;

extern float alonc, alatc;

extern float model_time, time0;

extern int iext;

extern float ramp;

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

extern intPtr i_global, j_global;

extern intPtr i_global_coarse, j_global_coarse;

extern floatPtr vabs, vabn, eln, els;
extern floatPtr uabw, uabe, elw, ele;

extern floatPtr dzz, dz, zz, z;

extern float xstart[], ystart[];

extern float (*fsm)[i_size], (*art)[i_size], 
			 (*aru)[i_size], (*arv)[i_size], 
			 (*dx)[i_size], (*dy)[i_size];

extern float (*east_e)[i_size], (*east_c)[i_size], 
			 (*east_u)[i_size], (*east_v)[i_size],
			 (*north_e)[i_size], (*north_c)[i_size], 
			 (*north_u)[i_size], (*north_v)[i_size], 
			 (*rot)[i_size];

extern float (*h)[i_size], (*dum)[i_size], (*dvm)[i_size], 
			 (*cor)[i_size];

extern float (*sbn)[i_size], (*sbs)[i_size], 
			 (*tbn)[i_size], (*tbs)[i_size],
			 (*vbn)[i_size], (*vbs)[i_size];

extern float (*tbe)[j_size], (*tbw)[j_size], 
			 (*sbe)[j_size], (*sbw)[j_size], 
			 (*ube)[j_size], (*ubw)[j_size];

extern float (*wubot)[i_size], (*wvbot)[i_size],  
			 (*uab)[i_size], (*vab)[i_size],
			 (*elb)[i_size], (*etb)[i_size], 
			 (*vfluxb)[i_size], (*d)[i_size], 
			 (*dt)[i_size], (*el)[i_size],
			 (*et)[i_size];

extern float (*wusurf)[i_size], (*wvsurf)[i_size], 
			 (*e_atmos)[i_size], (*vfluxf)[i_size], 
			 (*wtsurf)[i_size], (*swrad)[i_size], 
			 (*wssurf)[i_size];

extern float (*drx2d)[i_size], (*dry2d)[i_size]; 
			 
			 
extern float (*ub)[j_size][i_size], 
			 (*vb)[j_size][i_size]; 
			 

extern float (*uwsrf)[i_size], (*vwsrf)[i_size],
	         (*aamfac)[i_size]; 

extern float (*ustks)[j_size][i_size], 
			 (*vstks)[j_size][i_size],
			 (*xstks)[j_size][i_size],
			 (*ystks)[j_size][i_size];

extern float (*trb)[k_size][j_size][i_size],
			 (*tr)[k_size][j_size][i_size];


extern float (*tre)[k_size][j_size],
			 (*trw)[k_size][j_size],
 	  		 (*trn)[k_size][i_size],
 	  		 (*trs)[k_size][i_size];

extern float (*t)[j_size][i_size], (*s)[j_size][i_size], 
			 (*rho)[j_size][i_size], 
			 (*tb)[j_size][i_size], (*sb)[j_size][i_size];

extern float (*rmean)[j_size][i_size], 
			 (*tclim)[j_size][i_size], 
			 (*sclim)[j_size][i_size];

extern float (*sbn)[i_size], (*sbs)[i_size], 
			 (*tbn)[i_size], (*tbs)[i_size];

extern float (*tobw)[j_size][nfw],(*sobw)[j_size][nfw],
			 (*tobe)[j_size][nfe],(*sobe)[j_size][nfe],
			 (*tobs)[nfs][i_size],(*sobs)[nfs][i_size],
			 (*tobn)[nfn][i_size],(*sobn)[nfn][i_size];

extern float (*frz)[i_size];//it is a formal parameter

extern float (*ampe)[ntide], (*phae)[ntide],
			 (*amue)[ntide], (*phue)[ntide];


extern float (*ua)[i_size], (*va)[i_size],
			 (*elf)[i_size], (*etf)[i_size], 
			 (*uaf)[i_size], (*vaf)[i_size];
			 
extern float (*w)[j_size][i_size], 
			 (*l)[j_size][i_size], 
			 (*q2b)[j_size][i_size], 
			 (*q2)[j_size][i_size], 
			 (*q2lb)[j_size][i_size], 
			 (*q2l)[j_size][i_size], 
			 (*kh)[j_size][i_size], 
			 (*km)[j_size][i_size], 
			 (*kq)[j_size][i_size], 
			 (*aam)[j_size][i_size], 
			 (*u)[j_size][i_size], 
			 (*v)[j_size][i_size], 
			 (*drhox)[j_size][i_size], 
			 (*drhoy)[j_size][i_size]; 

extern float (*advx)[j_size][i_size], (*advy)[j_size][i_size];

extern float (*cbc)[i_size], (*aamfrz)[i_size];

extern float (*adx2d)[i_size], (*ady2d)[i_size], 
			 (*aam2d)[i_size], (*egf)[i_size], 
			 (*utf)[i_size], (*vtf)[i_size], 
			 (*advua)[i_size], (*advva)[i_size];

//!lyo:vort:
extern float (*fx)[i_size], (*fy)[i_size],
			 (*ctsurf)[i_size], (*ctbot)[i_size],
			 (*cpvf)[i_size], (*cjbar)[i_size],
			 (*cadv)[i_size], (*cten)[i_size],
			 (*totx)[i_size], (*toty)[i_size],
			 (*celg)[i_size], (*ctot)[i_size];

extern float (*utb)[i_size], (*vtb)[i_size]; //add by mode_internal
extern float (*uf)[j_size][i_size], 
			 (*vf)[j_size][i_size];//add by mode_internal
extern float (*tsurf)[i_size], (*ssurf)[i_size];//add by mode_internal
extern float (*tr3d)[j_size][i_size], 
			 (*tr3db)[j_size][i_size];//add by mode_internal
extern float (*rdisp2d)[i_size];//add by mode_internal
extern float (*egb)[i_size];//add by mode_internal
extern float (*wr)[j_size][i_size];//add by mode_internal
extern int inb;//add by mode_internal

extern float (*uab_mean)[i_size],//add by store_mean
			 (*vab_mean)[i_size],//add by store_mean
			 (*elb_mean)[i_size],//add by store_mean
			 (*wusurf_mean)[i_size],//add by store_mean
			 (*wvsurf_mean)[i_size],//add by store_mean
			 (*wtsurf_mean)[i_size],//add by store_mean
			 (*wssurf_mean)[i_size];//add by store_mean

extern float (*u_mean)[j_size][i_size], 
			 (*v_mean)[j_size][i_size],//add by store_mean
			 (*w_mean)[j_size][i_size],//add by store_mean
			 (*t_mean)[j_size][i_size],//add by store_mean
			 (*s_mean)[j_size][i_size],//add by store_mean
			 (*rho_mean)[j_size][i_size],//add by store_mean
			 (*kh_mean)[j_size][i_size],//add by store_mean
			 (*km_mean)[j_size][i_size];//add by store_mean

extern int num;//add by store_mean

extern float (*usrf_mean)[i_size],//add by store_surf_mean
			 (*vsrf_mean)[i_size],//add by store_surf_mean
			 (*elsrf_mean)[i_size],//add by store_surf_mean
			 (*uwsrf_mean)[i_size],//add by store_surf_mean
			 (*vwsrf_mean)[i_size],//add by store_surf_mean
			 (*utf_mean)[i_size],//add by store_surf_mean
			 (*vtf_mean)[i_size],//add by store_surf_mean
			 (*uwsrf)[i_size],//add by store_surf_mean
			 (*vwsrf)[i_size],//add by store_surf_mean
			 (*celg_mean)[i_size],//add by store_surf_mean
			 (*ctsurf_mean)[i_size],//add by store_surf_mean
			 (*cpvf_mean)[i_size],//add by store_surf_mean
			 (*cjbar_mean)[i_size],//add by store_surf_mean
			 (*cadv_mean)[i_size],//add by store_surf_mean
			 (*cten_mean)[i_size],//add by store_surf_mean
			 (*ctbot_mean)[i_size];//add by store_surf_mean

extern float (*xstks_mean)[j_size][i_size], 
			 (*ystks_mean)[j_size][i_size];//add by store_surf_mean

extern float (*relax_aid)[j_size][i_size];//added by mode_internal--advt1

extern float (*array_3d_tmp1)[j_size][i_size];//added by mode_internal--advt1
extern float (*array_3d_tmp2)[j_size][i_size];//added by mode_internal--advt1
extern float (*array_3d_tmp3)[j_size][i_size];//added by mode_internal--advt1

extern int nums;//add by store_surf_mean

extern int num_out;//add by store_surf_mean

extern char windf[4];
extern char title[40];
extern char netcdf_file[120];
extern char time_start[26];
extern char read_rst_file[120];
extern char write_rst_file[120];

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

/*
extern int n_east, n_west, n_north, n_south, 
		   my_task, master_task, pom_comm, jm_global;

extern int im, imm1, imm2, jm, jmm1, jmm2, kb, kbm1, kbm2;

extern int nitera, mode, ntp, iswtch;

extern int lramp, npg, ispadv, isplit, nbct, nbcs, nadv, iend;

extern float sw, dti2, tprni, grav, tbias, sbias, rhoref, umol, kappa, small, time0, dti, prtd2, period, horcon, ispi, isp2i, dte2, alpha, dte, 
			 rfw, rfe, rfs, rfn, smoth, vmaxl;

//extern int i_global[i_size], j_global[j_size];
extern intPtr i_global, j_global;

//extern float dzz[k_size], dz[k_size], zz[k_size], z[k_size];
extern floatPtr dzz, dz, zz, z;

//extern float uabw[j_size], elw[j_size], uabe[j_size], ele[j_size];
extern floatPtr uabw, elw, uabe, ele;

//extern float vabs[i_size], vabn[i_size], els[i_size], eln[i_size];
extern floatPtr vabs, vabn, els, eln;

//extern float fsm[j_size][i_size], art[j_size][i_size], 
//			 aru[j_size][i_size], arv[j_size][i_size], 
//			 dx[j_size][i_size], dy[j_size][i_size];

extern float (*fsm)[i_size], (*art)[i_size], 
			 (*aru)[i_size], (*arv)[i_size], 
			 (*dx)[i_size], (*dy)[i_size];

//extern float h[j_size][i_size], dum[j_size][i_size], dvm[j_size][i_size];
//extern float cbc[j_size][i_size], cor[j_size][i_size];
extern float (*h)[i_size], (*dum)[i_size], (*dvm)[i_size], 
			 (*cbc)[i_size], (*cor)[i_size];

//extern float tbe[k_size][j_size], tbw[k_size][j_size];
//extern float sbe[k_size][j_size], sbw[k_size][j_size];
//extern float ube[k_size][j_size], ubw[k_size][j_size];
extern float (*tbe)[j_size], (*tbw)[j_size], 
			 (*sbe)[j_size], (*sbw)[j_size], 
			 (*ube)[j_size], (*ubw)[j_size];

//extern float sbn[k_size][i_size], sbs[k_size][i_size];
//extern float tbn[k_size][i_size], tbs[k_size][i_size];
//extern float vbn[k_size][i_size], vbs[k_size][i_size];
extern float (*sbn)[i_size], (*sbs)[i_size], 
			 (*tbn)[i_size], (*tbs)[i_size],
			 (*vbn)[i_size], (*vbs)[i_size];

//extern float tsurf[j_size][i_size], ssurf[j_size][i_size];
extern float (*tsurf)[i_size], (*ssurf)[i_size];

//extern float rmean[k_size][j_size][i_size], tclim[k_size][j_size][i_size], sclim[k_size][j_size][i_size];
extern float (*rmean)[j_size][i_size], 
			 (*tclim)[j_size][i_size], 
			 (*sclim)[j_size][i_size];

//extern float aam_aid[j_size];//help calculate aam array in lateral_viscosity;
extern floatPtr aam_aid;

*/


/*
 * Below are variables assigned in initial process, 
 * and needed and changed in advance steps.
 * So we have to copy to c_functions when porting fortran to c
 */

//extern int iprint, iint;//interval in iint at which variables are printed


//--------------------------------------------------------
/*comments:
 *
 * model_time is used for replacing the "time" variable,
 * because "time" is a inherent variable in C++.
 *
 * When porting POM to GPU and using nvcc, 
 * compile will fail.
 *
 * In pure c_version code, it is not used.
 */

//extern float model_time;

//--------------------------------------------------------
/*comments:
 *
 * Below are just assigned 0 in initial.f, 
 * but assigned 0 again in surface_forcing in advance.f
 */
//extern float wusurf[j_size][i_size],wvsurf[j_size][i_size];
//extern float e_atmos[j_size][i_size],vfluxf[j_size][i_size];
//extern float wtsurf[j_size][i_size];
//extern float swrad[j_size][i_size], wssurf[j_size][i_size];

/*
extern float (*wusurf)[i_size], (*wvsurf)[i_size], 
			 (*e_atmos)[i_size], (*vfluxf)[i_size], 
			 (*wtsurf)[i_size], (*swrad)[i_size], 
			 (*wssurf)[i_size];
*/

//--------------------------------------------------------

//extern float dt[j_size][i_size];
//extern float wubot[j_size][i_size], wvbot[j_size][i_size];
//extern float d[j_size][i_size], ua[j_size][i_size];
//extern float va[j_size][i_size], uab[j_size][i_size];
//extern float vab[j_size][i_size], el[j_size][i_size];
//extern float elb[j_size][i_size]; 
//extern float egb[j_size][i_size], etb[j_size][i_size];
//extern float et[j_size][i_size];
//extern float utb[j_size][i_size], vtb[j_size][i_size];
//extern float vfluxb[j_size][i_size];
//

/*
extern float (*dt)[i_size], (*d)[i_size],
			 (*wubot)[i_size], (*wvbot)[i_size],  
			 (*ua)[i_size], (*va)[i_size], 
			 (*uab)[i_size], (*vab)[i_size],
			 (*el)[i_size], (*elb)[i_size], 
			 (*etb)[i_size], (*et)[i_size], 
			 (*utb)[i_size], (*vtb)[i_size], 
			 (*vfluxb)[i_size], (*egb)[i_size];
*/

//extern float w[k_size][j_size][i_size];
//extern float t[k_size][j_size][i_size], s[k_size][j_size][i_size];
//extern float v[k_size][j_size][i_size], u[k_size][j_size][i_size];
//extern float ub[k_size][j_size][i_size], aam[k_size][j_size][i_size];
//extern float vb[k_size][j_size][i_size];
//extern float rho[k_size][j_size][i_size];
//extern float drhox[k_size][j_size][i_size], drhoy[k_size][j_size][i_size];
//extern float q2[k_size][j_size][i_size],q2l[k_size][j_size][i_size];
//extern float q2b[k_size][j_size][i_size],q2lb[k_size][j_size][i_size];
//extern float tb[k_size][j_size][i_size],sb[k_size][j_size][i_size];

/*
extern float (*w)[j_size][i_size], (*t)[j_size][i_size], 
			 (*s)[j_size][i_size], (*v)[j_size][i_size], 
			 (*u)[j_size][i_size], (*ub)[j_size][i_size],
			 (*aam)[j_size][i_size], (*vb)[j_size][i_size], 
			 (*rho)[j_size][i_size], (*drhox)[j_size][i_size],
			 (*drhoy)[j_size][i_size], 
			 (*q2)[j_size][i_size], (*q2l)[j_size][i_size], 
			 (*q2b)[j_size][i_size], (*q2lb)[j_size][i_size],
			 (*tb)[j_size][i_size], (*sb)[j_size][i_size];
*/

//--------------------------------------------------------
//kq is only used in profq, but should be a global variable
//extern float kq[k_size][j_size][i_size];
//extern float (*kq)[j_size][i_size];
//--------------------------------------------------------

//extern float km[k_size][j_size][i_size];
//extern float kh[k_size][j_size][i_size];
//extern float (*km)[j_size][i_size], (*kh)[j_size][i_size];

//--------------------------------------------------------
//l is only modified and refernce in advance.f(proq_)
//extern float l[k_size][j_size][i_size]; 
//extern float (*l)[j_size][i_size];
//--------------------------------------------------------


/*
 * Below are variables only used in initial.f
 */

//in namelist &read_input
/*
extern char time_start[26];
extern char title[40];
extern char netcdf_file[120];
extern char read_rst_file[120];
extern char write_rst_file[120];

extern int nread_rst;
extern int irestart;

extern float write_rst;
extern float days;
extern float prtd1;
extern float prtd2;
extern float swtch;
extern float z0b;
extern float cbcmin;
extern float cbcmax;
extern float slmax;
extern float aam_init;
extern float pi;
*/

//extern float east_e[j_size][i_size], east_c[j_size][i_size],
//			 east_u[j_size][i_size], east_v[j_size][i_size];
//extern float north_e[j_size][i_size], north_c[j_size][i_size],
//			 north_u[j_size][i_size], north_v[j_size][i_size];
//
//extern float rot[j_size][i_size];

/*
extern float (*east_e)[i_size], (*east_c)[i_size], 
			 (*east_u)[i_size], (*east_v)[i_size],
			 (*north_e)[i_size], (*north_c)[i_size], 
			 (*north_u)[i_size], (*north_v)[i_size], 
			 (*rot)[i_size];
*/

//in distribute_mpi

/*
extern int n_proc;
extern int im_global, jm_global;
extern int im_local, jm_local;
*/


/*
 * Below are variables used in advance steps,
 * but not assigned in initial.f
 * So we don't need copy these variables from fortran to C
 * when porting
 */

/*
extern int error_status;
extern int iext;
//model time(days)
extern float time;
//inertial ramp 
extern float ramp;
*/

//extern float adx2d[j_size][i_size], ady2d[j_size][i_size];
//extern float drx2d[j_size][i_size], dry2d[j_size][i_size];
//extern float aam2d[j_size][i_size], egf[j_size][i_size];
//extern float utf[j_size][i_size], vtf[j_size][i_size];
//extern float advua[j_size][i_size], advva[j_size][i_size];
//extern float fluxua[j_size][i_size], fluxva[j_size][i_size];
//extern float elf[j_size][i_size], etf[j_size][i_size];
//extern float uaf[j_size][i_size], vaf[j_size][i_size];

/*
extern float (*adx2d)[i_size], (*ady2d)[i_size], 
			 (*drx2d)[i_size], (*dry2d)[i_size], 
			 (*aam2d)[i_size], (*egf)[i_size], 
			 (*utf)[i_size], (*vtf)[i_size], 
			 (*advua)[i_size], (*advva)[i_size], 
			 (*fluxua)[i_size], (*fluxva)[i_size],
			 (*elf)[i_size], (*etf)[i_size], 
			 (*uaf)[i_size], (*vaf)[i_size];
*/

//extern float advx[k_size][j_size][i_size];
//extern float advy[k_size][j_size][i_size];
//extern float uf[k_size][j_size][i_size];
//extern float vf[k_size][j_size][i_size];
//extern float (*advx)[j_size][i_size], (*advy)[j_size][i_size],			   
//			   (*uf)[j_size][i_size], (*vf)[j_size][i_size];

#endif
