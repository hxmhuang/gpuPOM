#include"data.h"

/*
 * below are const variables during advance steps
 */
MPI_Comm pom_comm, pom_comm_coarse;
int origin_task;
int my_task, 
    master_task, error_status, n_proc,
    im_global, im_local, jm_global, jm_local,
    im, imm1, imm2, jm, jmm1, jmm2, kb, kbm1, kbm2,
    n_east, n_west, n_north, n_south;

int im_coarse, im_local_coarse, im_global_coarse;
int jm_coarse, jm_local_coarse, jm_global_coarse;
int x_division, y_division;
int iprints;
struct tm dtime0, dtime;

int mode, nadv, nitera, npg, dte, isplit, nread_rst,
    iperx, ipery, n1d, ngrid;

int nse, nsw, nsn, nss; 

int lramp, ntp, nbct, nbcs, lspadv, iend, iprint, irestart,
    ispadv, iint;

int nb = 1, np;

int calc_wind, calc_tsforce, calc_river, calc_assim,
    calc_assimdrf, calc_tsurf_mc, calc_tide, calc_trajdrf,
    tracer_flag, calc_stokes, calc_vort, output_flag,
    SURF_flag,
    calc_interp; //!fhx:interp_flag

int iout, iouts, ioutv;

float rhoref, tbias, sbias, grav, kappa, z0b,
 	 cbcmin, cbcmax, horcon, tprni, umol, vmaxl,
 	 slmax, smoth, alpha, aam_init, pi, dti, dti2, dte2,
 	 isp2i, small, ispi, sw;

float rfw, rfe, rfs, rfn; 

float days, prtd1, prtd2, write_rst;

float lono = 999.0f, 
	  lato = 999.0f, 
	  xs = 1.5f, 
	  ys = 1.5f, 
	  fak = 0.5f;

float alonc, alatc;

float model_time, time0;

int iext;

float ramp;
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

intPtr i_global, j_global;

intPtr i_global_coarse, j_global_coarse;

floatPtr vabs, vabn, eln, els;
floatPtr uabw, uabe, elw, ele;

floatPtr dzz, dz, zz, z;

float xstart[100], ystart[100];

float (*fsm)[i_size], (*art)[i_size], 
	  (*aru)[i_size], (*arv)[i_size], 
	  (*dx)[i_size], (*dy)[i_size];

float (*east_e)[i_size], (*east_c)[i_size], 
	  (*east_u)[i_size], (*east_v)[i_size],
	  (*north_e)[i_size], (*north_c)[i_size], 
	  (*north_u)[i_size], (*north_v)[i_size], 
	  (*rot)[i_size];

float (*h)[i_size], (*dum)[i_size], (*dvm)[i_size], 
	  (*cor)[i_size];

float (*sbn)[i_size], (*sbs)[i_size], 
 	  (*tbn)[i_size], (*tbs)[i_size],
 	  (*vbn)[i_size], (*vbs)[i_size];

float (*tbe)[j_size], (*tbw)[j_size], 
 	  (*sbe)[j_size], (*sbw)[j_size], 
 	  (*ube)[j_size], (*ubw)[j_size];

float (*wubot)[i_size], (*wvbot)[i_size],  
 	  (*uab)[i_size], (*vab)[i_size],
 	  (*elb)[i_size], (*etb)[i_size], 
	  (*vfluxb)[i_size], (*d)[i_size], 
	  (*dt)[i_size], (*el)[i_size],
	  (*et)[i_size];

float (*wusurf)[i_size], (*wvsurf)[i_size], 
 	 (*e_atmos)[i_size], (*vfluxf)[i_size], 
 	 (*wtsurf)[i_size], (*swrad)[i_size], 
 	 (*wssurf)[i_size];

float (*drx2d)[i_size], (*dry2d)[i_size]; 
 	 
 	 
float (*ub)[j_size][i_size], 
 	  (*vb)[j_size][i_size]; 
 	 

float (*uwsrf)[i_size], (*vwsrf)[i_size],
      (*aamfac)[i_size]; 

float (*ustks)[j_size][i_size], 
 	 (*vstks)[j_size][i_size],
 	 (*xstks)[j_size][i_size],
 	 (*ystks)[j_size][i_size];

float (*trb)[k_size][j_size][i_size],
 	 (*tr)[k_size][j_size][i_size];

float (*tre)[k_size][j_size],
 	  (*trw)[k_size][j_size],
 	  (*trn)[k_size][i_size],
 	  (*trs)[k_size][i_size];

float (*t)[j_size][i_size], (*s)[j_size][i_size], 
 	  (*rho)[j_size][i_size], 
 	  (*tb)[j_size][i_size], (*sb)[j_size][i_size];

float (*rmean)[j_size][i_size], 
 	  (*tclim)[j_size][i_size], 
 	  (*sclim)[j_size][i_size];

float (*sbn)[i_size], (*sbs)[i_size], 
 	  (*tbn)[i_size], (*tbs)[i_size];

float (*tobw)[j_size][nfw],(*sobw)[j_size][nfw],
	  (*tobe)[j_size][nfe],(*sobe)[j_size][nfe],
	  (*tobs)[nfs][i_size],(*sobs)[nfs][i_size],
	  (*tobn)[nfn][i_size],(*sobn)[nfn][i_size];

float (*frz)[i_size];
float (*ampe)[ntide], (*phae)[ntide],
	  (*amue)[ntide], (*phue)[ntide];

float (*ua)[i_size], (*va)[i_size],
	  (*elf)[i_size], (*etf)[i_size], 
	  (*uaf)[i_size], (*vaf)[i_size];
			 
float (*w)[j_size][i_size], 
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

float (*advx)[j_size][i_size], (*advy)[j_size][i_size];

float (*cbc)[i_size], (*aamfrz)[i_size];

float (*adx2d)[i_size], (*ady2d)[i_size], 
	  (*aam2d)[i_size], (*egf)[i_size], 
	  (*utf)[i_size], (*vtf)[i_size], 
	  (*advua)[i_size], (*advva)[i_size];

//!lyo:vort:
float (*fx)[i_size], (*fy)[i_size],
	  (*ctsurf)[i_size], (*ctbot)[i_size],
	  (*cpvf)[i_size], (*cjbar)[i_size],
	  (*cadv)[i_size], (*cten)[i_size],
	  (*totx)[i_size], (*toty)[i_size],
	  (*celg)[i_size], (*ctot)[i_size];

float (*utb)[i_size], (*vtb)[i_size]; //add by mode_internal
float (*uf)[j_size][i_size], 
 	 (*vf)[j_size][i_size];//add by mode_internal
float (*tsurf)[i_size], (*ssurf)[i_size];//add by mode_internal
float (*tr3d)[j_size][i_size], 
 	 (*tr3db)[j_size][i_size];//add by mode_internal
float (*rdisp2d)[i_size];//add by mode_internal
float (*egb)[i_size];//add by mode_internal
float (*wr)[j_size][i_size]; 
int inb;//add by mode_internal


float (*uab_mean)[i_size],//add by store_mean
 	 (*vab_mean)[i_size],//add by store_mean
 	 (*elb_mean)[i_size],//add by store_mean
 	 (*wusurf_mean)[i_size],//add by store_mean
 	 (*wvsurf_mean)[i_size],//add by store_mean
 	 (*wtsurf_mean)[i_size],//add by store_mean
 	 (*wssurf_mean)[i_size];//add by store_mean

float (*u_mean)[j_size][i_size], 
 	 (*v_mean)[j_size][i_size],//add by store_mean
 	 (*w_mean)[j_size][i_size],//add by store_mean
 	 (*t_mean)[j_size][i_size],//add by store_mean
 	 (*s_mean)[j_size][i_size],//add by store_mean
 	 (*rho_mean)[j_size][i_size],//add by store_mean
 	 (*kh_mean)[j_size][i_size],//add by store_mean
 	 (*km_mean)[j_size][i_size];//add by store_mean

int num;//add by store_mean

float (*usrf_mean)[i_size],//add by store_surf_mean
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

float (*xstks_mean)[j_size][i_size], 
 	 (*ystks_mean)[j_size][i_size];//add by store_surf_mean

float (*relax_aid)[j_size][i_size];//added by mode_internal--advt1

float (*array_3d_tmp1)[j_size][i_size];//added by mode_internal--advt1
float (*array_3d_tmp2)[j_size][i_size];//added by mode_internal--advt1
float (*array_3d_tmp3)[j_size][i_size];//added by mode_internal--advt1

int nums;//add by store_surf_mean
int num_out=0;//add by store_surf_mean

char windf[4];
char title[40];
char netcdf_file[120];
char time_start[26];
char read_rst_file[120];
char write_rst_file[120];




void init_array(){
	i_global = (intPtr)malloc(i_size*sizeof(int));
	j_global = (intPtr)malloc(j_size*sizeof(int));
	i_global_coarse = (intPtr)malloc(i_coarse_size*sizeof(int));
	j_global_coarse = (intPtr)malloc(j_coarse_size*sizeof(int));

	vabs = (floatPtr)malloc(i_size*sizeof(float));
	vabn = (floatPtr)malloc(i_size*sizeof(float));
	eln = (floatPtr)malloc(i_size*sizeof(float));
	els = (floatPtr)malloc(i_size*sizeof(float));

	uabw = (floatPtr)malloc(j_size*sizeof(float));
	uabe = (floatPtr)malloc(j_size*sizeof(float));
	elw = (floatPtr)malloc(j_size*sizeof(float));
	ele = (floatPtr)malloc(j_size*sizeof(float));

	dzz = (floatPtr)malloc(k_size*sizeof(float));
	dz = (floatPtr)malloc(k_size*sizeof(float));
	zz = (floatPtr)malloc(k_size*sizeof(float));
	z = (floatPtr)malloc(k_size*sizeof(float));

	sbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	sbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	tbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	tbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	vbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	vbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));

	tbe = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	tbw = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	sbe = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	sbw = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	ube = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	ubw = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));

	wubot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wvbot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	uab = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vab = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	elb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	etb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vfluxb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	wusurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wvsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	e_atmos = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vfluxf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wtsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	swrad = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wssurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	drx2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dry2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	uwsrf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vwsrf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	aamfac = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	ub = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	vb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));

	ustks = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	vstks = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	xstks = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	ystks = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));

	tre = (float(*)[k_size][j_size])malloc(nb*k_size*j_size*sizeof(float));
	trw = (float(*)[k_size][j_size])malloc(nb*k_size*j_size*sizeof(float));
	trn = (float(*)[k_size][i_size])malloc(nb*k_size*i_size*sizeof(float));
	trs = (float(*)[k_size][i_size])malloc(nb*k_size*i_size*sizeof(float));

	trb = (float(*)[k_size][j_size][i_size])malloc(nb*k_size*j_size*i_size*sizeof(float));
	tr = (float(*)[k_size][j_size][i_size])malloc(nb*k_size*j_size*i_size*sizeof(float));

	
	fsm = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	art = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	aru = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	arv = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dx = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dy = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	
	east_e = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	east_c = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	east_u = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	east_v = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_e = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_c = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_u = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_v = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	rot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	h = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dum = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dvm = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cor = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dt = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	el = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	et = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	
	t = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	s = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	tb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	sb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	rho = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	rmean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	tclim = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	sclim = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	sbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	sbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	tbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	tbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));

	tobw = (float(*)[j_size][nfw])malloc(k_size*j_size*nfw*sizeof(float));
	sobw = (float(*)[j_size][nfw])malloc(k_size*j_size*nfw*sizeof(float));
	tobe = (float(*)[j_size][nfe])malloc(k_size*j_size*nfe*sizeof(float));
	sobe = (float(*)[j_size][nfe])malloc(k_size*j_size*nfe*sizeof(float));
	tobs = (float(*)[nfs][i_size])malloc(k_size*nfs*i_size*sizeof(float));
	sobs = (float(*)[nfs][i_size])malloc(k_size*nfs*i_size*sizeof(float));
	tobn = (float(*)[nfn][i_size])malloc(k_size*nfn*i_size*sizeof(float));
	sobn = (float(*)[nfn][i_size])malloc(k_size*nfn*i_size*sizeof(float));

	frz = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	ampe = (float(*)[ntide])malloc(j_size*ntide*sizeof(float));
	phae = (float(*)[ntide])malloc(j_size*ntide*sizeof(float));
	amue = (float(*)[ntide])malloc(j_size*ntide*sizeof(float));
	phue = (float(*)[ntide])malloc(j_size*ntide*sizeof(float));
	
	
	ua = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	va = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	etf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	elf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	uaf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vaf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	
	w = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	l = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2b = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2 = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2lb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2l = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	kh = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	km = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	kq = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	aam = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	u = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	v = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	drhox = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	drhoy = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	cbc = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	aamfrz = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	advx = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	advy = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));

	adx2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ady2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	aam2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	egf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	utf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vtf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	advua = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	advva = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));


	fx = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	fy = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ctsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ctbot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cpvf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cjbar = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cadv = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cten = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	totx = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	toty = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	celg = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ctot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	
	utb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vtb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	uf = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	vf = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	tsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ssurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	tr3d = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	tr3db = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	rdisp2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	egb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wr = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	uab_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vab_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	elb_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wusurf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wvsurf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wtsurf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wssurf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	u_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	v_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	w_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	t_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	s_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	rho_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	kh_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	km_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	usrf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vsrf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	elsrf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	uwsrf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vwsrf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	utf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vtf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	uwsrf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vwsrf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	celg_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ctsurf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cpvf_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cjbar_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cadv_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cten_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ctbot_mean = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	xstks_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	ystks_mean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	relax_aid = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	array_3d_tmp1 = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	array_3d_tmp2 = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	array_3d_tmp3 = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));

}

void assign_aid_array(){
	int i, j, k;
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				relax_aid[k][j][i]	= 1.586e-8f*(1.f-expf(zz[k]*h[j][i]*5.e-4f));
			}
		}
	}
}
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
int n_east, n_west, n_north, n_south, 
    my_task, master_task, pom_comm, jm_global;

int im, imm1, imm2, jm, jmm1, jmm2, kb, kbm1, kbm2;

int nitera, mode, ntp, iswtch;

int lramp, npg, ispadv, isplit, nbct, nbcs, nadv, iend;

float sw, dti2, tprni, grav, tbias, sbias,
 	 rhoref, umol, kappa, small, time0, dti, prtd2, 
 	 period, horcon, ispi, isp2i, dte2, alpha, dte, 
 	 rfw, rfe, rfs, rfn, smoth, vmaxl;

//int i_global[i_size], j_global[j_size];
intPtr i_global, j_global;

//float dzz[k_size], dz[k_size], zz[k_size], z[k_size];
floatPtr dzz, dz, zz, z;

//float uabw[j_size], elw[j_size], uabe[j_size], ele[j_size];
floatPtr uabw, elw, uabe, ele;

//float vabs[i_size], vabn[i_size], els[i_size], eln[i_size];
floatPtr vabs, vabn, els, eln;

//float fsm[j_size][i_size], art[j_size][i_size], 
// 	  aru[j_size][i_size], arv[j_size][i_size], 
// 	  dx[j_size][i_size], dy[j_size][i_size];
float (*fsm)[i_size], (*art)[i_size], 
	  (*aru)[i_size], (*arv)[i_size], 
	  (*dx)[i_size], (*dy)[i_size];

//float h[j_size][i_size], 
//	  dum[j_size][i_size], 
//	  dvm[j_size][i_size];
//
//float cbc[j_size][i_size], 
//	  cor[j_size][i_size];
float (*h)[i_size], (*dum)[i_size], (*dvm)[i_size], 
	  (*cbc)[i_size], (*cor)[i_size];

//float tbe[k_size][j_size], tbw[k_size][j_size];
//float sbe[k_size][j_size], sbw[k_size][j_size];
//float ube[k_size][j_size], ubw[k_size][j_size];
float (*tbe)[j_size], (*tbw)[j_size], 
	  (*sbe)[j_size], (*sbw)[j_size], 
	  (*ube)[j_size], (*ubw)[j_size];

//float sbn[k_size][i_size], sbs[k_size][i_size];
//float tbn[k_size][i_size], tbs[k_size][i_size];
//float vbn[k_size][i_size], vbs[k_size][i_size];
float (*sbn)[i_size], (*sbs)[i_size], 
	  (*tbn)[i_size], (*tbs)[i_size],
	  (*vbn)[i_size], (*vbs)[i_size];

//float tsurf[j_size][i_size], ssurf[j_size][i_size];
float (*tsurf)[i_size], (*ssurf)[i_size];


//float rmean[k_size][j_size][i_size], 
// 	  tclim[k_size][j_size][i_size], 
// 	  sclim[k_size][j_size][i_size];
float (*rmean)[j_size][i_size], 
	  (*tclim)[j_size][i_size], 
	  (*sclim)[j_size][i_size];

*/

/*comments:
 *
 * aam_aid helps calculate aam array in lateral_viscosity.
 *
 * It is mainly for solving the problem of 
 * expf function difference bewteen CPU and GPU.
 *
 * In pure c_version code, it is not used. 
 */
//float aam_aid[j_size];
//floatPtr aam_aid;

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////


/*
 * Below are variables assigned in initial process, 
 * and needed and changed in advance steps.
 *
 * So we have to copy to c_functions when porting fortran to c
 */

//int iprint, iint;//interval in iint at which variables are printed

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
//float model_time;
//--------------------------------------------------------
/*comments:
 *
 * Below are just assigned 0 in initial.f, 
 * but assigned 0 again in surface_forcing in advance.f
 */
//float wusurf[j_size][i_size],wvsurf[j_size][i_size];
//float e_atmos[j_size][i_size],vfluxf[j_size][i_size];
//float wtsurf[j_size][i_size];
//float swrad[j_size][i_size], wssurf[j_size][i_size];

/*
float (*wusurf)[i_size], (*wvsurf)[i_size], 
	  (*e_atmos)[i_size], (*vfluxf)[i_size], 
	  (*wtsurf)[i_size], (*swrad)[i_size], 
	  (*wssurf)[i_size];
*/

//--------------------------------------------------------

//float dt[j_size][i_size];
//float wubot[j_size][i_size], wvbot[j_size][i_size];
//float d[j_size][i_size], ua[j_size][i_size];
//float va[j_size][i_size], uab[j_size][i_size];
//float vab[j_size][i_size], el[j_size][i_size];
//float elb[j_size][i_size]; 
//float egb[j_size][i_size], etb[j_size][i_size];
//float et[j_size][i_size];
//float utb[j_size][i_size], vtb[j_size][i_size];
//float vfluxb[j_size][i_size];

/*
float (*dt)[i_size], (*d)[i_size],
	  (*wubot)[i_size], (*wvbot)[i_size],  
	  (*ua)[i_size], (*va)[i_size], 
	  (*uab)[i_size], (*vab)[i_size],
	  (*el)[i_size], (*elb)[i_size], 
	  (*etb)[i_size], (*et)[i_size], 
	  (*utb)[i_size], (*vtb)[i_size], 
	  (*vfluxb)[i_size], (*egb)[i_size];
*/

//float w[k_size][j_size][i_size];
//float t[k_size][j_size][i_size], s[k_size][j_size][i_size];
//float v[k_size][j_size][i_size], u[k_size][j_size][i_size];
//float ub[k_size][j_size][i_size], 
//		aam[k_size][j_size][i_size];
//float vb[k_size][j_size][i_size];
//float rho[k_size][j_size][i_size];
//float drhox[k_size][j_size][i_size], 
//		drhoy[k_size][j_size][i_size];
//float q2[k_size][j_size][i_size],
//		q2l[k_size][j_size][i_size];
//float q2b[k_size][j_size][i_size],
//		q2lb[k_size][j_size][i_size];
//float tb[k_size][j_size][i_size],
//	    sb[k_size][j_size][i_size];

/*
float (*w)[j_size][i_size], (*t)[j_size][i_size], 
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
//float kq[k_size][j_size][i_size];
//float (*kq)[j_size][i_size];
//--------------------------------------------------------

//float km[k_size][j_size][i_size];
//float kh[k_size][j_size][i_size];
//float (*km)[j_size][i_size], (*kh)[j_size][i_size];

//--------------------------------------------------------
//l is only modified and refernce in advance.f(proq_)
//float l[k_size][j_size][i_size]; 
//float (*l)[j_size][i_size];
//--------------------------------------------------------

/*
 * Below are variables only used in initial.f
 */

//in namelist &read_input

/*
char time_start[26];
char title[40];
char netcdf_file[120];
char read_rst_file[120];
char write_rst_file[120];
float write_rst;
float days;
float prtd1;
float prtd2;
float swtch;
int nread_rst;
float z0b;
float cbcmin;
float cbcmax;
float slmax;
float aam_init;
float pi;
int irestart;
*/

//float east_e[j_size][i_size], east_c[j_size][i_size],
//	  east_u[j_size][i_size], east_v[j_size][i_size];
//float north_e[j_size][i_size], north_c[j_size][i_size],
//	  north_u[j_size][i_size], north_v[j_size][i_size];
//float rot[j_size][i_size];

/*
float (*east_e)[i_size], (*east_c)[i_size], 
	  (*east_u)[i_size], (*east_v)[i_size],
	  (*north_e)[i_size], (*north_c)[i_size], 
	  (*north_u)[i_size], (*north_v)[i_size], 
	  (*rot)[i_size];
*/

//in distribute_mpi

/*
int n_proc;
int im_global, jm_global;
int im_local, jm_local;
*/



/*
 * Below are variables used in advance steps,
 * but not assigned in initial.f
 * So we don't need copy these variables from fortran to C
 * when porting
 */
/*
int error_status;
int iext;
//model time(days)
float time;
//inertial ramp 
float ramp;
*/

//float adx2d[j_size][i_size], ady2d[j_size][i_size];
//float drx2d[j_size][i_size], dry2d[j_size][i_size];
//float aam2d[j_size][i_size], egf[j_size][i_size];
//float utf[j_size][i_size], vtf[j_size][i_size];
//float advua[j_size][i_size], advva[j_size][i_size];
//float fluxua[j_size][i_size], fluxva[j_size][i_size];
//float elf[j_size][i_size], etf[j_size][i_size];
//float uaf[j_size][i_size], vaf[j_size][i_size];

/*
float (*adx2d)[i_size], (*ady2d)[i_size], 
	  (*drx2d)[i_size], (*dry2d)[i_size], 
	  (*aam2d)[i_size], (*egf)[i_size], 
	  (*utf)[i_size], (*vtf)[i_size], 
	  (*advua)[i_size], (*advva)[i_size], 
	  (*fluxua)[i_size], (*fluxva)[i_size],
	  (*elf)[i_size], (*etf)[i_size], 
	  (*uaf)[i_size], (*vaf)[i_size];
*/

//float advx[k_size][j_size][i_size];
//float advy[k_size][j_size][i_size];
//float uf[k_size][j_size][i_size];
//float vf[k_size][j_size][i_size];

/*
float (*advx)[j_size][i_size], (*advy)[j_size][i_size],		
	  (*uf)[j_size][i_size], (*vf)[j_size][i_size];
*/


/*
void init_array(){
	i_global = (intPtr)malloc(i_size*sizeof(int));
	j_global = (intPtr)malloc(j_size*sizeof(int));

	dzz = (floatPtr)malloc(k_size*sizeof(float));
	dz = (floatPtr)malloc(k_size*sizeof(float));
	zz = (floatPtr)malloc(k_size*sizeof(float));
	z = (floatPtr)malloc(k_size*sizeof(float));

	uabw = (floatPtr)malloc(j_size*sizeof(float));
	uabe = (floatPtr)malloc(j_size*sizeof(float));
	elw = (floatPtr)malloc(j_size*sizeof(float));
	ele = (floatPtr)malloc(j_size*sizeof(float));

	vabs = (floatPtr)malloc(i_size*sizeof(float));
	vabn = (floatPtr)malloc(i_size*sizeof(float));
	els = (floatPtr)malloc(i_size*sizeof(float));
	eln = (floatPtr)malloc(i_size*sizeof(float));

	aam_aid = (floatPtr)malloc(j_size*sizeof(float));
	//////////////////////////////////////////////////////
	//2D	
	fsm = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	art = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	aru = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	arv = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dx = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dy = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	h = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dum = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dvm = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cbc = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	cor = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	tbe = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	tbw = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	sbe = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	sbw = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	ube = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));
	ubw = (float(*)[j_size])malloc(k_size*j_size*sizeof(float));

	sbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	sbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	tbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	tbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	vbn = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));
	vbs = (float(*)[i_size])malloc(k_size*i_size*sizeof(float));

	tsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ssurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	//////////////////////////////////////////////////////
	//3D	
	rmean = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	tclim = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	sclim = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	//////////////////////////////////////////////////////
	//2D	
	wusurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wvsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	e_atmos = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vfluxf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wtsurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	swrad = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wssurf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	dt = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wubot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	wvbot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ua = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	va = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	uab = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vab = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	el = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	elb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	egb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	etb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	et = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	utb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vtb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vfluxb = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	//////////////////////////////////////////////////////
	//3D	
	w = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	t = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	s = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	v = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	u = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	ub = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	aam = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	vb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	rho = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	drhox = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	drhoy = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2 = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2l = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2b = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	q2lb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	tb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	sb = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	kq = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	km = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	kh = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	l = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));


	//////////////////////////////////////////////////////
	//2D	
	east_e = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	east_c = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	east_u = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	east_v = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_e = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_c = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_u = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	north_v = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	rot = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	
	adx2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	ady2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	drx2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	dry2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	aam2d = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	egf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	utf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vtf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	advua = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	advva = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	fluxua = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	fluxva = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	elf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	etf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	uaf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));
	vaf = (float(*)[i_size])malloc(j_size*i_size*sizeof(float));

	//////////////////////////////////////////////////////
	//3D	
	advx = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	advy = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	uf = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
	vf = (float(*)[j_size][i_size])malloc(k_size*j_size*i_size*sizeof(float));
		
}
*/
