#include<stdio.h>
#include<string.h>
#include<unistd.h>

#include"cinitialize.h"
#include"csolver.h"


void initialize(){

	initialize_mpi();

	read_input();

	distribute_mpi();

	distribute_mpi_coarse();

	if ((im_global_coarse-2)*x_division == (im_global-2) &&
		(jm_global_coarse-2)*y_division == (jm_global-2))
		calc_interp = 1;
	else if (im_global_coarse == im_global &&
			 jm_global_coarse == jm_global)
		calc_interp = 0;
	else{
		if (my_task == master_task){
			printf("Incompatible number of *_global and *_global_coarse\n"
				   "POM terminated with error\n");
			exit(1);
		}
	}

	initialize_arrays();

	read_grid();

	initial_conditions();

	lateral_boundary_conditions();

	if (calc_trajdrf)
		read_trajdrf();

	bfrz(nfw, nfe, nfs, nfn,
		 n_west, n_east, n_south, n_north,
		 im, jm, 6, frz);

	int i, j;
	float rdisp = dti/86400.f;

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			frz[j][i] *= rdisp;		
		}
	}

	bfrz(nsw, nse, nss, nsn,
		 n_west, n_east, n_south, n_north,
		 im, jm, 6, aamfrz);

	rdisp = 4.f;

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			aamfrz[j][i] *= rdisp;		
		}
	}
	
	if (calc_tide)
		read_tide();
	
	update_initial();


	bottom_friction();

	if (lono != 999.f && lato != 999.f)
		incmix(aamfac, im, jm, 1, east_e, north_e);


	sum0i_mpi(&error_status, master_task);
	bcast0d_mpi(&error_status, master_task);
	

	if (error_status != 0){
		if (my_task == master_task)
			printf("error_status = %d\n"
				   "POM Terminated With Error\n", error_status);

		finalize_mpi();
		exit(1);
	}

	if (my_task == master_task){
		printf("Initializaiton ends successfully\n");	
	}

	//wind_init();

	return;
}

/*
void read_input_(char *f_title, char *f_netcdf_file,
				 int *f_mode, int *f_nadv, int *f_nitera,
				 float *f_sw, int *f_npg, float *f_dte,
				 int *f_isplit, char *f_time_start,
				 int *f_nread_rst, char *f_read_rst_file,
				 float *f_write_rst, char *f_write_rst_file,
				 float *f_days, float *f_prtd1, float *f_prtd2,
				 int *f_iperx, int *f_ipery, int *f_n1d,
				 int *f_ngrid, char *f_windf,
				 int *f_calc_wind, int *f_calc_tsforce,
				 int *f_calc_river, int *f_calc_assim,
				 int *f_calc_assimdrf, int *f_calc_tsurf_mc,
				 int *f_calc_tide, int *f_calc_trajdrf,
				 int *f_tracer_flag, int *f_calc_stokes,
				 int *f_calc_vort, int *f_output_flag,
				 int *f_SURF_flag,

				 int *f_lramp, float *f_rhoref,
				 float *f_tbias, float *f_sbias, float *f_grav,
				 float *f_kappa, float *f_z0b, float *f_cbcmin,
				 float *f_cbcmax, float *f_horcon, float *f_tprni,
				 float *f_umol, float *f_vmaxl, float *f_slmax,
				 int *f_ntp, int *f_nbct, int *f_nbcs, int *f_ispadv,
				 float *f_smoth, float *f_alpha, float *f_aam_init,
				 int *f_nse, int *f_nsw, int *f_nsn, int *f_nss, 
				 float *f_small, float *f_pi, float *f_dti, float *f_dte2,
				 float *f_dti2, int *f_iend, int *f_iprint, 
				 int *f_irestart, int *f_iprints, 
				 float *f_ispi, float *f_isp2i, 
				 float *f_time0, float *f_time){
*/

void read_input(){
	
//! read input namelist
	read_namelist_pom_(title,      netcdf_file, 
					   &mode,      &nadv,
				       &nitera,	   &sw,
				       &npg,	   &dte,
				       &isplit,    time_start,
				       &nread_rst, read_rst_file,
				       &write_rst, write_rst_file,
				       &days,      &prtd1,     
				       &prtd2,     &iperx,
				       &ipery,     &n1d,
				       &ngrid,     windf,
				       &im_global, &jm_global,
				       &kb,        &im_local,
				       &jm_local,  &im_global_coarse,
				       &jm_global_coarse, &im_local_coarse,
				       &jm_local_coarse, &x_division,
				       &y_division, &n_proc);

//! read main switches
	read_namelist_switch_(&calc_wind, &calc_tsforce,
						  &calc_river, &calc_assim,
						  &calc_assimdrf, &calc_tsurf_mc,
						  &calc_tide, &tracer_flag,
						  &calc_stokes, &calc_vort,
						  &output_flag, &SURF_flag);

	if (ngrid == -1){
		mode = 2; dte = 10.0f; isplit = 45;
	}

	if (ngrid == -2){
		mode = 3; iperx = 1; dte = 10.0; //isplit = 45;	
	}

// Logical for inertial ramp (.true. if inertial ramp to be applied
// to wind stress and baroclinic forcing, otherwise .false.)
    //lramp=false;
    lramp=0;

// Reference density (recommended values: 1025 for seawater,
// 1000 for freswater; S.I. units):
    rhoref=1025.e0f;

// Temperature bias (deg. C)
    tbias=0.e0f;

// Salinity bias

    sbias=0.e0f;
// gravity constant (S.I. units)

    grav=9.806e0f;
// von Karman's constant
    kappa=0.4e0f;

// Bottom roughness (metres)
    z0b=.01e0f;

// Minimum bottom friction coeff.
    cbcmin=.00015e0f;

// Maximum bottom friction coeff.
    cbcmax=cbcmin;

// Smagorinsky diffusivity coeff.
    horcon=0.05e0f;

// Inverse horizontal turbulent Prandtl number (ah/am; dimensionless):
// NOTE that tprni=0.e0 yields zero horizontal diffusivity!
    tprni=.2e0f;

// Background viscosity used in subroutines profq, proft, profu and
// profv (S.I. units)
    umol=2.e-5f;

//! Maximum magnitude of vaf (used in check that essentially tests
//! for CFL violation):
    vmaxl=5.e0f;
	
// Maximum allowable value of:
//   <difference of depths>/<sum of depths>
// for two adjacent cells (dimensionless). This is used in subroutine
// slpmax. If >= 1, then slpmax is not applied:
    slmax=2.e0f;

// Water type, used in subroutine proft.
//    ntp    Jerlov water type
//     1            i
//     2            ia
//     3            ib
//     4            ii
//     5            iii
    ntp=2;

// Surface temperature boundary condition, used in subroutine proft:
//    nbct   prescribed    prescribed   short wave
//           temperature      flux      penetration
//     1        no           yes           no
//     2        no           yes           yes
//     3        yes          no            no
//     4        yes          no            yes
    nbct=1;

// Surface salinity boundary condition, used in subroutine proft:
//    nbcs   prescribed    prescribed
//            salinity      flux
//     1        no           yes
//     3        yes          no
// NOTE that only 1 and 3 are allowed for salinity.
    nbcs=1;

// Step interval during which external (2-D) mode advective terms are
// not updated (dimensionless):
    ispadv=5;

// Constant in temporal filter used to prevent solution splitting
// (dimensionless):
    smoth=0.10e0f;

// Weight used for surface slope term in external (2-D) dynamic
// equation (a value of alpha = 0.e0 is perfectly acceptable, but the
// value, alpha=.225e0 permits a longer time step):
    alpha=0.225e0f;

// Initial value of aam:
    aam_init=10.e0f;

//! Sponges: boundary zones where aam & cbc are increased:
      nse=0; nsw=0; nsn=0; nss=0;

// End of input of constants

// calculate some constants
    small=1.e-10f;            // Small value

    pi=atanf(1.e0f)*4.e0f;    // PI

    dti=dte*(float)isplit;

    dte2=dte*2;

    dti2=dti*2;

    iend=MAX(NINT(days*24.e0*3600.e0/dti),2);

    iprint=NINT(prtd1*24.e0*3600.e0/dti);

    irestart=NINT(write_rst*24.e0*3600.e0/dti);

    iprints=NINT(prtd2*24.e0*3600.e0/dti);

    ispi=1.e0f/(float)isplit;

    isp2i=1.e0f/(2.e0f*(float)isplit);

//! initialise time
//! Calulate the Julian days from 1992-01-01 !fhx:20110113
	strptime(time_start, "%Y-%m-%d_%H:%M:%S", &dtime0);
	strptime(read_rst_file, "restart.%Y-%m-%d_%H_%M_%S.nc", &dtime);

	time0 = (mktime(&dtime0)-mktime(&dtime))/86400;

//    time0=0.e0;
    model_time=time0;

	if (my_task == master_task){
		printf("%-15s=\t%s\n", "title", title);	
		printf("%-15s=\t%d\n", "mode", mode);	
		printf("%-15s=\t%d\n", "nadv", nadv);	
		printf("%-15s=\t%d\n", "nitera", nitera);	
		printf("%-15s=\t%f\n", "sw", sw);	
		printf("%-15s=\t%d\n", "npg", npg);	//!fhx:Toni:npg
		printf("%-15s=\t%d\n", "nread_rst", nread_rst);	
		printf("%-15s=\t%f\n", "write_rst", write_rst);	
		printf("%-15s=\t%d\n", "irestart", irestart);	
		printf("%-15s=\t%f\n", "dte", dte);	
		printf("%-15s=\t%f\n", "dti", dti);	
		printf("%-15s=\t%d\n", "isplit", isplit);	
		printf("%-15s=\t%s\n", "time_start", time_start);	
		printf("%-15s=\t%f\n", "days", days);	
		printf("%-15s=\t%f\n", "time0", time0);	
		printf("%-15s=\t%d\n", "iend", iend);	
		printf("%-15s=\t%f\n", "prtd1", prtd1);	
		printf("%-15s=\t%d\n", "iprint", iprint);	
		printf("%-15s=\t%f\n", "prtd2", prtd2);	
		printf("%-15s=\t%f\n", "rhoref", rhoref);	
		printf("%-15s=\t%f\n", "tbias", tbias);	
		printf("%-15s=\t%f\n", "sbias", sbias);	
		printf("%-15s=\t%f\n", "grav", grav);	
		printf("%-15s=\t%f\n", "kappa", kappa);	
		printf("%-15s=\t%f\n", "z0b", z0b);	
		printf("%-15s=\t%f\n", "cbcmin", cbcmin);	
		printf("%-15s=\t%f\n", "cbcmax", cbcmax);	
		printf("%-15s=\t%f\n", "horcon", horcon);	
		printf("%-15s=\t%f\n", "tprni", tprni);	
		printf("%-15s=\t%f\n", "umol", umol);	
		printf("%-15s=\t%f\n", "vmaxl", vmaxl);	
		printf("%-15s=\t%f\n", "slmax", slmax);	
		printf("%-15s, %-15s=\t%f, %f\n", "lono", "lato", lono, lato);	
		printf("%-15s, %-15s, %-15s=\t%f, %f, %f\n", 
				"xs", "ys", "fak", xs, ys, fak);	
		printf("%-15s, %-15s=\t%d, %d\n", "iperx", "ipery", iperx, ipery);	
		printf("%-15s=\t%d\n", "ntp", ntp);	
		printf("%-15s=\t%d\n", "nbct", nbct);	
		printf("%-15s=\t%d\n", "nbcs", nbcs);	
		printf("%-15s=\t%d\n", "ispadv", ispadv);	
		printf("%-15s=\t%f\n", "smoth", smoth);	
		printf("%-15s=\t%f\n", "alpha", alpha);	
		printf("%-15s=\t%d\n", "n1d", n1d);	
		printf("%-15s=\t%d\n", "ngrid", ngrid);	
		printf("%-15s=\t%d\n", "lramp", lramp);	
		printf("%-15s=\t%d\n", "calc_wind", calc_wind);	
		printf("%-15s=\t%d\n", "calc_tsforce", calc_tsforce);	
		printf("%-15s=\t%d\n", "calc_river", calc_river);	
		printf("%-15s=\t%d\n", "calc_assim", calc_assim);	
		printf("%-15s=\t%d\n", "calc_assimdrf", calc_assimdrf);	
		printf("%-15s=\t%d\n", "calc_tsurf_mc", calc_tsurf_mc);	
		printf("%-15s=\t%d\n", "calc_tide", calc_tide);	
		printf("%-15s=\t%d\n", "calc_trajdrf", calc_trajdrf);	
		printf("%-15s=\t%d\n", "calc_stokes", calc_stokes);	
		printf("%-15s=\t%d\n", "calc_vort", calc_vort);	
		printf("%-15s=\t%d\n", "calc_interp", calc_interp);	
		printf("%-15s=\t%d\n", "tracer_flag", tracer_flag);	
		printf("%-15s=\t%d\n", "output_flag", output_flag);	
		printf("%-15s=\t%d\n", "SURF_flag", SURF_flag);	
	}

// print initial summary	

	//pom-nml
	/*
	strcpy(f_title, title); 
	strcpy(f_netcdf_file, netcdf_file);
	*f_mode = mode;
	*f_nadv = nadv;
	*f_nitera = nitera;
	*f_sw = sw;
	*f_npg = npg;
	*f_dte = dte;
	*f_isplit = isplit;
	strcpy(f_time_start, time_start);
	*f_nread_rst = nread_rst;
	strcpy(f_read_rst_file, read_rst_file);
	*f_write_rst = write_rst;
	strcpy(f_write_rst_file, write_rst_file);
	*f_days = days;
	*f_prtd1 = prtd1;
	*f_prtd2 = prtd2;
	*f_iperx = iperx;
	*f_ipery = ipery;
	*f_n1d = n1d;
	*f_ngrid = ngrid;
	strcpy(f_windf, windf);

	//switch-nml
	*f_calc_wind = calc_wind;
	*f_calc_tsforce = calc_tsforce;
	*f_calc_river = calc_river;
	*f_calc_assim = calc_assim;
	*f_calc_assimdrf = calc_assimdrf;
	*f_calc_tsurf_mc = calc_tsurf_mc;
	*f_calc_tide = calc_tide;
	*f_calc_trajdrf = calc_trajdrf;
	*f_tracer_flag = tracer_flag;
	*f_calc_stokes = calc_stokes;
	*f_calc_vort = calc_vort;
	*f_output_flag = output_flag;
	*f_SURF_flag = SURF_flag;

	*f_lramp = lramp;
	*f_rhoref = rhoref;
	*f_tbias = tbias;
	*f_sbias = sbias;
	*f_grav = grav;
	*f_kappa = kappa;
	*f_z0b = z0b;
	*f_cbcmin = cbcmin;
	*f_cbcmax = cbcmax;
	*f_horcon = horcon;
	*f_tprni = tprni;
	*f_umol = umol;
	*f_vmaxl = vmaxl;
	*f_slmax = slmax;
	*f_ntp = ntp;
	*f_nbct = nbct;
	*f_nbcs = nbcs;
	*f_ispadv = ispadv;
	*f_smoth = smoth;
	*f_alpha = alpha;
	*f_aam_init = aam_init;
	*f_nse = nse;
	*f_nsw = nsw;
	*f_nsn = nsn;
	*f_nss = nss;

	*f_small = small;
	*f_pi = pi;
	*f_dti = dti;
	*f_dte2 = dte2;
	*f_dti2 = dti2;

	*f_iend = iend;
	*f_iprint = iprint;
	*f_irestart = irestart;
	*f_iprints = iprints;

	*f_ispi = ispi;
	*f_isp2i = isp2i;
	*f_time0 = time0;
	*f_time = model_time;
	*/
	
    return;
}

/*
void initialize_arrays_(float *f_vabn, float *f_vabs, 
					    float *f_eln, float *f_els,
					    float f_vbn[][i_size], float f_vbs[][i_size],
					    float f_tbn[][i_size], float f_tbs[][i_size],
					    float f_sbn[][i_size], float f_sbs[][i_size],
					    float *f_uabe, float *f_uabw, 
					    float *f_ele, float *f_elw,
					    float f_ube[][j_size], float f_ubw[][j_size],
					    float f_tbe[][j_size], float f_tbw[][j_size],
					    float f_sbe[][j_size], float f_sbw[][j_size],
					    float f_uab[][j_size], float f_vab[][j_size],
					    float f_elb[][j_size], float f_etb[][j_size],
					    float f_e_atmos[][j_size], float f_vfluxb[][j_size],
					    float f_vfluxf[][j_size], 
						float f_wusurf[][j_size], float f_wvsurf[][j_size],
						float f_wtsurf[][j_size], float f_wssurf[][j_size],
						float f_swrad[][j_size], 
						float f_drx2d[][j_size], float f_dry2d[][j_size],
						float f_uwsrf[][j_size], float f_vwsrf[][j_size],
					    float f_wubot[][j_size], float f_wvbot[][j_size],
					    float f_aamfac[][j_size], 
						float f_ub[][j_size][i_size],
						float f_vb[][j_size][i_size],
						float f_ustks[][j_size][i_size],
						float f_vstks[][j_size][i_size],
						float f_xstks[][j_size][i_size],
						float f_ystks[][j_size][i_size],

						float f_trb[][k_size][j_size][i_size],
						float f_tr[][k_size][j_size][i_size],
						
						float f_tre[][j_size][i_size],
						float f_trw[][j_size][i_size],
						float f_trn[][j_size][i_size],
						float f_trs[][j_size][i_size]){
*/

void initialize_arrays(){

	int i, j, k, m;

	iout = 0;
	iouts = 0;
	ioutv = 0;

	for (i = 0; i < im; i++){
		vabn[i] = 0;
		vabs[i] = 0;
		eln[i] = 0;
		els[i] = 0;

		/*
		f_vabn[i] = vabn[i];
		f_vabs[i] = vabs[i];
		f_eln[i] = eln[i];
		f_els[i] = els[i];
		*/
	}

	for (k = 0; k < kb; k++){
		for (i = 0; i < im; i++){
			vbn[k][i] = 0;
			vbs[k][i] = 0;
			tbn[k][i] = 0;
			tbs[k][i] = 0;
			sbn[k][i] = 0;
			sbs[k][i] = 0;


			/*
			f_vbn[k][i] = vbn[k][i];
			f_vbs[k][i] = vbs[k][i];
			f_tbn[k][i] = tbn[k][i];
			f_tbs[k][i] = tbs[k][i];
			f_sbn[k][i] = sbn[k][i];
			f_sbs[k][i] = sbs[k][i];
			*/
		}
	}

	for (j = 0; j < jm; j++){
		uabe[j] = 0;
		uabw[j] = 0;
		ele[j] = 0;
		elw[j] = 0;


		/*
		f_uabe[j] = uabe[j];
		f_uabw[j] = uabw[j];
		f_ele[j] = ele[j];
		f_elw[j] = elw[j];
		*/
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			ube[k][j] = 0;
			ubw[k][j] = 0;
			tbe[k][j] = 0;
			tbw[k][j] = 0;
			sbe[k][j] = 0;
			sbw[k][j] = 0;

			
			/*
			f_ube[k][j] = ube[k][j];
			f_ubw[k][j] = ubw[k][j];
			f_tbe[k][j] = tbe[k][j];
			f_tbw[k][j] = tbw[k][j];
			f_sbe[k][j] = sbe[k][j];
			f_sbw[k][j] = sbw[k][j];
			*/
		}
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			uab[j][i] = 0;
			vab[j][i] = 0;
			elb[j][i] = 0;
			etb[j][i] = 0;
			e_atmos[j][i] = 0;
			vfluxb[j][i] = 0;
			vfluxf[j][i] = 0;
			wusurf[j][i] = 0;
			wvsurf[j][i] = 0;
			wtsurf[j][i] = 0;
			wssurf[j][i] = 0;
			swrad[j][i] = 0;
			drx2d[j][i] = 0;
			dry2d[j][i] = 0;
			uwsrf[j][i] = 0;
			vwsrf[j][i] = 0;
			wubot[j][i] = 0;
			wvbot[j][i] = 0;
			aamfac[j][i] = 0;

			/*
			f_uab[j][i] = uab[j][i];
			f_vab[j][i] = vab[j][i];
			f_elb[j][i] = elb[j][i];
			f_etb[j][i] = etb[j][i];
			f_e_atmos[j][i] = e_atmos[j][i];
			f_vfluxb[j][i] = vfluxb[j][i];
			f_vfluxf[j][i] = vfluxf[j][i];
			f_wusurf[j][i] = wusurf[j][i];
			f_wvsurf[j][i] = wvsurf[j][i];
			f_wtsurf[j][i] = wtsurf[j][i];
			f_wssurf[j][i] = wssurf[j][i];
			f_swrad[j][i] = swrad[j][i];
			f_drx2d[j][i] = drx2d[j][i];
			f_dry2d[j][i] = dry2d[j][i];
			f_uwsrf[j][i] = uwsrf[j][i];
			f_vwsrf[j][i] = vwsrf[j][i];
			f_wubot[j][i] = wubot[j][i];
			f_wvbot[j][i] = wvbot[j][i];
			f_aamfac[j][i] = aamfac[j][i];
			*/
		}
	}

	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				ub[k][j][i] = 0;
				vb[k][j][i] = 0;
				ustks[k][j][i] = 0;
				vstks[k][j][i] = 0;
				xstks[k][j][i] = 0;
				ystks[k][j][i] = 0;
				

				/*
				f_ub[k][j][i] = ub[k][j][i];
				f_vb[k][j][i] = vb[k][j][i];
				f_ustks[k][j][i] = ustks[k][j][i];
				f_vstks[k][j][i] = vstks[k][j][i];
				f_xstks[k][j][i] = xstks[k][j][i];
				f_ystks[k][j][i] = ystks[k][j][i];
				*/
			}
		}
	}

	for (m = 0; m < nb; m++){
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					trb[m][k][j][i] = 0;
					tr[m][k][j][i] = 0;

					/*
					f_trb[m][k][j][i] = trb[m][k][j][i];
					f_tr[m][k][j][i] = tr[m][k][j][i];
					*/
				}
			}
		}
	}

	for (m = 0; m < nb; m++){
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				tre[m][k][j] = 0;
				trw[m][k][j] = 0;

				/*
				f_tre[m][k][j] = tre[m][k][j];
				f_trw[m][k][j] = trw[m][k][j];
				*/
			}

			for (i = 0; i < im; i++){
				trn[m][k][i] = 0;
				trs[m][k][i] = 0;

				/*
				f_trn[m][k][i] = trn[m][k][i];
				f_trs[m][k][i] = trs[m][k][i];
				*/
			}
		}
	}
	return;
}


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
				float f_north_v[][i_size]){
*/
void read_grid(){

	int i, j, k;
	float rdisp;
	float deg2rad;

	deg2rad = pi/180;

	if (ngrid <= 0){
		specify_grid();
	}else{
		read_grid_pnetcdf();
	}
	
//!
//! Artificially increase dx & dy for 1d-simulations
	if (n1d != 0){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				dx[j][i] = 1.e9f;	
				dy[j][i] = 1.e9f;
			}
		}
	}

	for (k = 0; k < kb-1; k++){
		dz[k] = z[k]-z[k+1];	
		dzz[k] = zz[k]-zz[k+1];
	}

	dz[kb-1] = dz[kb-2];
	dzz[kb-1] = dzz[kb-2];

	if (my_task == master_task){
		printf("%3s %7s %10s %10s %10s\n", "k", "z", "zz", "dz", "dzz");
		for (k = 0; k < kb; k++){
			printf("%3d %10f %10f %10f %10f\n", k, z[k], zz[k], dz[k], dzz[k]);		
		}
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			cor[j][i] = 2.0f*7.29e-5f*sinf(north_e[j][i]*deg2rad);
		}
	}

    rdisp=2.0f*7.29e-5f*sinf(24.3f*deg2rad);

	if (ngrid == -1){
	//!topograhic standing wave in an f-plane channel:
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				cor[j][i] = rdisp;	
			}
		}
	}else if (ngrid == -2){
	//!dambreak (cold-warm) problem in f-plane channel
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				cor[j][i] = rdisp;	
			}
		}
	}

//! calculate areas of "t" and "s" cells
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			art[j][i] = dx[j][i]*dy[j][i];	
		}
	}
//! calculate areas of "u" and "v" cells
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			aru[j][i] = 0.25f*(dx[j][i]+dx[j][i-1])*(dy[j][i]+dy[j][i-1]);
			arv[j][i] = 0.25f*(dx[j][i]+dx[j-1][i])*(dy[j][i]+dy[j-1][i]);
		}
	}

	//exchange2d_mpi_xsz_(aru, im, jm);
	exchange2d_mpi(aru, im, jm);
	//exchange2d_mpi_xsz_(arv, im, jm);
	exchange2d_mpi(arv, im, jm);

	if (n_west == -1){
		for (j = 0; j < jm; j++){
			aru[j][0] = aru[j][1];	
			arv[j][0] = arv[j][1];
		}
	}

	if (n_south == -1){
		for (i = 0; i < im; i++){
			aru[0][i] = aru[1][i];
			arv[0][i] = arv[1][i];
		}
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i< im; i++){
			d[j][i] = h[j][i] + el[j][i];	
			dt[j][i] = h[j][i] + et[j][i];	
		}
	}


	/*
	*f_nsw= nsw;
	*f_nse= nse;
	*f_nss= nss;
	*f_nsn= nsn;
	*f_alonc = alonc;
	*f_alatc = alatc;

	for (k = 0; k < kb; k++){
		f_z[k] = z[k];
		f_zz[k] = zz[k];
		f_dz[k] = dz[k];
		f_dzz[k] = dzz[k];
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_cor[j][i] = cor[j][i];
			f_art[j][i] = art[j][i];
			f_aru[j][i] = aru[j][i];
			f_arv[j][i] = arv[j][i];
			f_d[j][i] = d[j][i];
			f_dt[j][i] = dt[j][i];
			f_east_u[j][i] = east_u[j][i];	
			f_east_v[j][i] = east_v[j][i];	
			f_east_e[j][i] = east_e[j][i];	
			f_east_c[j][i] = east_c[j][i];	
			f_north_u[j][i] = north_u[j][i];	
			f_north_v[j][i] = north_v[j][i];	
			f_north_e[j][i] = north_e[j][i];	
			f_north_c[j][i] = north_c[j][i];	
			f_rot[j][i] = rot[j][i];	
			f_dx[j][i] = dx[j][i];	
			f_dy[j][i] = dy[j][i];	
			f_h[j][i] = h[j][i];	
			f_fsm[j][i] = fsm[j][i];	
			f_dum[j][i] = dum[j][i];	
			f_dvm[j][i] = dvm[j][i];	
		}
	}
	*/
}


//35 parameters
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
				  float f_north_v[][i_size]){
*/

void specify_grid(){

	float rad = 0.01745329f; 
	float re = 6371.e3f;
	float deg2rad = 1.745329e-2f; 
	float awest, asouth, del;
	int logornolog;
	float dlon, dlat, xc, yc, rdisp;
	int i, j, k;
	float alon[j_global_size][i_global_size], 
		  alat[j_global_size][i_global_size];

	k = im_global;

	if (ngrid == -1){
		awest = 100.0f;	
		asouth = 20.5f;
		del = 0.1f;
		logornolog = 0;
		nsw = 0;
		nse = 45;
		nss = 0;
		nsn = 0;
		k = im_global/2;
	}else if (ngrid == -2){
		awest = 100.0f;	
		asouth = 20.5f;
		del = 0.01f;
		logornolog = 0;
		nsw = 0;
		nse = 0;
		nss = 0;
		nsn = 0;
	}

	//modify -z, -zz, //-dz, -dzz 
	depth(z, zz, kb, logornolog);

	for (j = 0; j < jm_global; j++){
		for (i = 0; i < im_global; i++){
			if (i == 0){
				alon[j][i] = awest;	
			}else{
				if ((i+1) <= k)
					alon[j][i] = alon[j][i-1]+del;
				else
					alon[j][i] = alon[j][i-1]+del*(1.0f+((i+1)-k)*0.025f);
			}

			if (((i+1) >= i_global[0] && (i+1) <= i_global[im-1]) &&
				((j+1) >= j_global[0] && (j+1) <= j_global[jm-1]))
				east_e[j-j_global[0]+1][i-i_global[0]+1] = alon[j][i];
		}
	}

	for (j = 0; j < jm_global; j++){
		for (i = 0; i < im_global; i++){
			if (j == 0){
				alat[j][i] = asouth;	
			}else{
				alat[j][i] = alat[j-1][i] + del;	
			}

			if (((i+1) >= i_global[0] && (i+1) <= i_global[im-1]) &&
				((j+1) >= j_global[0]-1 && (j+1) <= j_global[jm-1]))
				north_e[j-j_global[0]+1][i-i_global[0]+1] = alat[j][i];
		}
	}

	//exchange2d_mpi(east_e, im, jm);
	exchange2d_mpi(north_e, im, jm);

	//modify -east_c, -north_c -east_u, -north_u, -east_v, -north_v
	//reference east_e, north_e
	east_north_ecuv();

	for (j = 0; j < jm; j++){
		for (i = 0; i < im-1; i++){
			rot[j][i] = 0;	
			dlat = north_e[j][i+1] - north_e[j][i];
			dlon = east_e[j][i+1] - east_e[j][i];
			if (dlon != 0)
				rot[j][i] = atanf(dlat/dlon);
		}
		rot[j][im-1] = rot[j][im-2];
	}

	exchange2d_mpi(east_c, im, jm);
	exchange2d_mpi(north_c, im, jm);
	exchange2d_mpi(east_u, im, jm);
	exchange2d_mpi(north_u, im, jm);
	exchange2d_mpi(east_v, im, jm);
	exchange2d_mpi(north_v, im, jm);
	exchange2d_mpi(rot, im, jm);

	for (j = 1; j < jm-1; j++){
		for (i = 1; i < im-1; i++){
			float tmp_eu = (east_e[j][i+1] - east_e[j][i-1])
					       *cosf(north_e[j][i]*rad);
			float tmp_nu = (north_e[j][i+1]-north_e[j][i-1]);
			float tmp_ev = (east_e[j+1][i]-east_e[j-1][i])
						   *cosf(north_e[j][i]*rad);
			float tmp_nv = (north_e[j+1][i]-north_e[j-1][i]);
				
			dx[j][i] = 0.5f*rad*re*sqrtf((tmp_eu*tmp_eu)+(tmp_nu*tmp_nu));
			dy[j][i] = 0.5f*rad*re*sqrtf((tmp_ev*tmp_ev)+(tmp_nv*tmp_nv));
		}
	}

	for (i = 0; i < im; i++){
		dx[0][i] = dx[1][i];	
		dx[jm-1][i] = dx[jm-2][i];
		dy[0][i] = dy[1][i];
		dy[jm-1][i] = dy[jm-2][i];
	}

	for (j = 0; j < jm; j++){
		dx[j][0] = dx[j][1];	
		dx[j][im-1] = dx[j][im-2];
		dy[j][0] = dy[j][1];
		dy[j][im-1] = dy[j][im-2];
	}

	exchange2d_mpi(dx, im, jm);
	exchange2d_mpi(dy, im, jm);

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			h[j][i] = 50.0f;	
		}
	}

	if (ngrid == -1){//!topograhic standing wave in a channel:
		for (i = 0; i < im; i++){
			for (j = 0; j < jm; j++){
				//!jm_glo*=47
				h[j][i] = 84.0f-(1.6e-4f)*(north_e[j][i]-asouth)*deg2rad*re;
			}
		}
		xc = alon[6][49];
		yc = alat[6][49];

		for (i = 0; i < im; i++){
			for (j = 0; j < jm; j++){
				float tmp_e = (east_e[j][i]-xc)/0.5f;
				float tmp_n = (north_e[j][i]-yc)/1.0f;
				rdisp = 75.0f*(1.0f-0.66f*expf(-(tmp_e*tmp_e+tmp_n*tmp_n)));
				if (rdisp < h[j][i])
					h[j][i] = rdisp;
			}
		}
//! Put land in SW region - Taiwan Island:
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				if (north_e[j][i] <= 21.0f){
					if (east_e[j][i] <= 107.2f){
						h[j][i] = 1.0f;	
					}
				}
			}
		}

//! Insert a zonal wall at mid-channel
		xc = 101.2f;
		yc = alat[25][im_global/2-1];

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				if (north_e[j][i] <= yc+0.5f*del &&
					north_e[j][i] >= yc){
					if (east_e[j][i] >= xc && 
						east_e[j][i] <= xc+6.8f){
						h[j][i] = 1.0f;	
					}
				}
			}
		}
	}

//! Put walls all around:
	if (n_north == -1){
		for (i = 0; i < im; i++){
			h[jm-1][i] = 1.0f;	
		}
	}
	if (n_south == -1){
		for (i = 0; i < im; i++){
			h[0][i] = 1.0f;
		}
	}
	if (n_east == -1){
		for (j = 0; j < jm; j++){
			h[j][im-1] = 1.0f;	
		}
	}
	if (n_west == -1){
		for (j = 0; j < jm; j++){
			h[j][0] = 1.0f;	
		}
	}

	if (ngrid == -2){//!west-east periodic
		if (n_east == -1){
			for (j = 0; j < jm; j++){
				h[j][im-1] = h[j][im-2];	
			}
		}
		if (n_west == -1){
			for (j = 0; j < jm; j++){
				h[j][0] = h[j][1];	
			}
		}
	}

	exchange2d_mpi(h, im, jm);

	for (i = 0; i < im; i++){
		for (j = 0; j < jm; j++){
			if (h[j][i] > 1.0f){
				fsm[j][i] = 1.0f;
			}else{
				fsm[j][i] = 0;	
			}
		}
	}

	for (i = 1; i < im; i++){
		for (j = 0; j < jm; j++){
			dum[j][i] = fsm[j][i]*fsm[j][i-1];	
		}
	}

	for (j = 0; j < jm; j++){
		dum[j][0] = dum[j][1];	
	}

	for (j = 1; j < jm; j++){
		for (i = 0; i < im; i++){
			dvm[j][i] = fsm[j][i] * fsm[j-1][i];	
		}
	}

	for (i = 0; i < im; i++){
		dvm[0][i] = dvm[1][i];	
	}

	exchange2d_mpi(fsm, im, jm);
	exchange2d_mpi(dum, im, jm);
	exchange2d_mpi(dvm, im, jm);

//! Center lon/lat:
	
	alonc = alon[jm_global/2-1][im_global/2-1];
	alatc = alat[jm_global/2-1][im_global/2-1];

	/*
	*f_nsw = nsw;
	*f_nse = nse;
	*f_nss = nss;
	*f_nsn = nsn;

	*f_alonc = alonc;
	*f_alatc = alatc;

	for (k = 0; k < kb; k++){
		f_z[k] = z[k];
		f_zz[k] = zz[k];
	}



	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_east_e[j][i] = east_e[j][i];
			f_north_e[j][i] = north_e[j][i];
			f_rot[j][i] = rot[j][i];
			f_dx[j][i] = dx[j][i];
			f_dy[j][i] = dy[j][i];
			f_h[j][i] = h[j][i];
			f_fsm[j][i] = fsm[j][i];
			f_dum[j][i] = dum[j][i];
			f_dvm[j][i] = dvm[j][i];

			f_east_c[j][i] = east_c[j][i];
			f_north_c[j][i] = north_c[j][i];
			f_east_u[j][i] = east_u[j][i];
			f_north_u[j][i] = north_u[j][i];
			f_east_v[j][i] = east_v[j][i];
			f_north_v[j][i] = north_v[j][i];
		}
	}
	*/

	return;
}


/***********************************************************************
 *                                                                    *
 * FUNCTION    :  Establishes the vertical sigma grid with log        *
 *                distributions at the top and bottom and a linear    *
 *                distribution in between. The number of layers of    *
 *                reduced thickness are kl1-2 at the surface and      *
 *                kb-kl2-1 at the bottom. kl1 and kl2 are defined in  *
 *                the main program. For no log portions, set kl1=2    *
 *                and kl2=kb-1.                                       *
 *                                                                    *
 **********************************************************************/
void depth(float *z, float *zz,
		   int kb, int logornolog){

	/*
	int kb = *f_kb;
	int logornolog = *f_logornolog;
	*/
	

	float delz;	
	int kdz[12] = {1, 1, 2, 4,
				   8, 16, 32, 64,
				   128, 256, 512, 1024};
	int k, kl1, kl2;

	/*
	for (k = 0; k < kb; k++){
		z[k] = f_z[k];
		zz[k] = f_zz[k];
		dz[k] = f_dz[k];
		dzz[k] = f_dzz[k];
	}
	*/

	if (logornolog != 0){		//!logarithmic top and/or bottom
		kl1 = 6; kl2 = kb-5;	//!logornolog=1 !logarithmic top and bottom
		if (logornolog == 2)    //!logarithmic top only
			kl2 = kb-1;
		if (logornolog == 3)	//!logarithmic bottom only
			kl1 = 2;
	}else{						//!logornolog=0 !uniformly-spaced sigma levels
		kl1 = 2; kl2 = kb-1;	
	}

	z[0] = 0;

	for (k = 1; k < kl1; k++){
		z[k] = z[k-1]+kdz[k-1];	
	}

	delz = z[kl1-1] - z[kl1-2];

	for (k = kl1; k < kl2; k++){
		z[k] = z[k-1]+delz;	
	}

	for (k = kl2; k < kb; k++){
		dz[k] = ((float)(kdz[kb-k-1]))*delz/((float)(kdz[kb-kl2-1]));
		z[k] = z[k-1]+dz[k];
	}

	for (k = 0; k < kb; k++){
		z[k] = -z[k]/z[kb-1];	
	}

	for (k = 0; k < kb-1; k++){
		zz[k] = 0.5f*(z[k]+z[k+1]);
	}

	zz[kb-1] = 2.0f*zz[kb-2] - zz[kb-3];

	/*
	for (k = 0; k < kb-1; k++){
		dz[k] = z[k] - z[k+1];	
		dzz[k] = zz[k] - zz[k+1];
	}

	dz[kb-1] = 0;
	dzz[kb-1] = 0;
	*/
	return;
}

void east_north_ecuv(){
	int i, j;	
//! Corner of cell points:
	
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			east_c[j][i] = (east_e[j][i] + east_e[j][i-1]
						   +east_e[j-1][i] + east_e[j-1][i-1])/4.0f;
			north_c[j][i] = (north_e[j][i] + north_e[j][i-1]
						    +north_e[j-1][i] + north_e[j-1][i-1])/4.0f;
		}
	}

//! Extrapolate ends (approx.):

	for (i = 1; i < im; i++){
		east_c[0][i] = 2.0f*east_c[1][i]-east_c[2][i];	
		north_c[0][i] = 2.0f*north_c[1][i]-north_c[2][i];
	}

	east_c[0][0] = 2.0f*east_c[0][1]-east_c[0][2];

	for (j = 1; j < jm; j++){
		east_c[j][0] = 2.0f*east_c[j][1] - east_c[j][2];	
		north_c[j][0] = 2.0f*north_c[j][1] - north_c[j][2];
	}

	north_c[0][0] = 2.0f*north_c[1][0] - north_c[2][0];

//! u-points:
	for (j = 0; j < jm-1; j++){
		for (i = 0; i < im; i++){
			east_u[j][i] = (east_c[j][i] + east_c[j+1][i])/2.0f;
			north_u[j][i] = (north_c[j][i] + north_c[j+1][i])/2.0f;
		}
	}

//! Extrapolate ends:
	
	for (i = 0; i < im; i++){
		east_u[jm-1][i]	= (east_c[jm-1][i]*3.0f - east_c[jm-2][i])/2.0f;
		north_u[jm-1][i] = (north_c[jm-1][i]*3.0f - north_c[jm-2][i])/2.0f;
	}
//! v-points:
	
	for (j = 0; j < jm; j++){
		for (i = 0; i < im-1; i++){
			east_v[j][i] = (east_c[j][i] + east_c[j][i+1])/2.0f;
			north_v[j][i] = (north_c[j][i] + north_c[j][i+1])/2.0f;
		}
	}

//! Extrapolate ends:
	
	for (j = 0; j < jm; j++){
		east_v[j][im-1] = (east_c[j][im-1]*3.0f - east_c[j][im-2])/2.0f;
		north_v[j][im-1] = (north_c[j][im-1]*3.0f - north_c[j][im-2])/2.0f;
	}

	return;

}

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
						 float f_sbs[][i_size]){
*/

void initial_conditions(){
	
	int i, j, k;
    int ii, jj;                        //!lyo:pac10:
    char netcdf_ic_file[120];		   //!lyo:20110202:
    int fexist;                        //!lyo:20110202:

	if (ngrid > 0){
//!
//! read initial temperature and salinity from ic file
		read_initial_ts_pnetcdf(kb, tb, sb);	

//! read annual-mean, xy-ave t,sclim if avail !lyo:20110202:
		strcpy(netcdf_ic_file, "./in/tsclim/ts_mean.nc");
		if (access(netcdf_ic_file, F_OK) == 0){
			//! annual-mean xy-ave t,sclim
			read_mean_ts_pnetcdf(tclim, sclim);	
		}else{
			//! annual-mean *clim* - dangerous for rmean
			read_clim_ts_pnetcdf(tclim, sclim);	
		}
	}else{
		//! ngrid=<0, user can specify T/S here:
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					tb[k][j][i] = 20.0f;
					sb[k][j][i] = 35.0f;
					tclim[k][j][i] = 20.0f;
					sclim[k][j][i] = 35.0f;
				}
			}
		}

		if (ngrid == -2){
			//! dambreak (cold-warm) problem in f-plane channel
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						if (north_e[j][i] >= alatc){
							tb[k][j][i] = 15.0f;//! Cooler north	
						}else{
							tb[k][j][i] = 25.0f;//! Warmer south
						}
						tclim[k][j][i] = 20.0f;
					}
				}
			}
		}
	}

//! calc. initial density
    dens(sb,tb,rho);
//! calc. rmean; should use xy-ave t,sclim; ok if tsforce is called later
    dens(sclim,tclim,rmean);

	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				t[k][j][i] = tb[k][j][i];	
				s[k][j][i] = sb[k][j][i];
			}
		}	
	}

//! set thermodynamic boundary conditions (for the seamount problem, and
//! other possible applications, lateral thermodynamic boundary conditions
//! are set equal to the initial conditions and are held constant
//! thereafter - users may create variable boundary conditions)
	
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < nfw; i++){
				tobw[k][j][i] = tclim[k][j][i]*fsm[j][i];	
				sobw[k][j][i] = sclim[k][j][i]*fsm[j][i];
			}

			for (i = 0; i < nfe; i++){
				int ii = im-i-1;	
				tobe[k][j][i] = tclim[k][j][ii]*fsm[j][ii];
				sobe[k][j][i] = sclim[k][j][ii]*fsm[j][ii];
			}
			
			tbw[k][j] = tobw[k][j][0];
			sbw[k][j] = sobw[k][j][0];
			tbe[k][j] = tobe[k][j][0];
			sbe[k][j] = sobe[k][j][0];
		}

		for (i = 0; i < im; i++){
			for (j = 0; j < nfs; j++){
				tobs[k][j][i] = tclim[k][j][i]*fsm[j][i];
				sobs[k][j][i] = sclim[k][j][i]*fsm[j][i];
			}

			for (j = 0; j < nfn; j++){
				int jj = jm-j-1;	
				tobn[k][j][i] = tclim[k][jj][i]*fsm[jj][i];
				sobn[k][j][i] = sclim[k][jj][i]*fsm[jj][i];
			}
			
			tbs[k][i] = tobs[k][0][i];
			sbs[k][i] = sobs[k][0][i];
			tbn[k][i] = tobn[k][0][i];
			sbn[k][i] = sobn[k][0][i];
		}
	}




	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				f_tb[k][j][i] = tb[k][j][i];		
				f_sb[k][j][i] = sb[k][j][i];		
				f_t[k][j][i] = t[k][j][i];		
				f_s[k][j][i] = s[k][j][i];		
				f_tclim[k][j][i] = tclim[k][j][i];		
				f_sclim[k][j][i] = sclim[k][j][i];		
				f_rho[k][j][i] = rho[k][j][i];		
				f_rmean[k][j][i] = rmean[k][j][i];		
			}
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < nfw; i++){
				f_tobw[k][j][i] = tobw[k][j][i];
				f_sobw[k][j][i] = sobw[k][j][i];
			}

			for (i = 0; i < nfe; i++){
				f_tobe[k][j][i] = tobe[k][j][i];
				f_sobe[k][j][i] = sobe[k][j][i];
			}
			
			f_tbw[k][j] = tbw[k][j];
			f_sbw[k][j] = sbw[k][j];
			f_tbe[k][j] = tbe[k][j];
			f_sbe[k][j] = sbe[k][j];
		}

		for (i = 0; i < im; i++){
			for (j = 0; j < nfs; j++){
				f_tobs[k][j][i] = tobs[k][j][i];
				f_sobs[k][j][i] = sobs[k][j][i];
			}

			for (j = 0; j < nfn; j++){
				f_tobn[k][j][i] = tobn[k][j][i];
				f_sobn[k][j][i] = sobn[k][j][i];
			}
			
			f_tbs[k][i] = tbs[k][i];
			f_sbs[k][i] = sbs[k][i];
			f_tbn[k][i] = tbn[k][i];
			f_sbn[k][i] = sbn[k][i];
		}
	}
	*/

}

/*
void lateral_boundary_conditions_(float *f_rfe, float *f_rfw,
							     float *f_rfn, float *f_rfs,
								 float f_cor[][i_size]){
*/
void lateral_boundary_conditions(){

	int here;
	int i, j, ic, jc;
	float corcon;

	read_uabe_pnetcdf();

//! Radiation factors for use in subroutine bcond !alu:20101216 
    rfe=1.e0; rfw=1.e0; rfn=1.e0; rfs=1.e0; //! =1 Flather; =0 clamped

//! Periodic in "x" and/or "y"?  !lyo:20110224:alu:stcc:
//!     iperx.ne.0 if x-periodic; ipery.ne.0 if y-periodic               !
//!     iperx(y) < 0 if south/north (west/east) walls are free-slip      !
//!     iperx=-1; ipery= 0 !--> x-periodic & free-slip S/N boundaries
//!     iperx= 0; ipery= 0 !--> x-periodic & free-slip S/N boundaries
//!     cannot be beta-plane if double-periodic (note cor(*,*) 
//!     was previously defined in "call read_grid")

	if (iperx == 1 && ipery == 1){
		ic = (im_global+1)/2;
		jc = (jm_global+1)/2;
		here = judge_inout(ic, jc, 
						   i_global[0], i_global[jm-1],
						   j_global[0], j_global[jm-1]);
		if (here){
			corcon = cor[jc-j_global[0]][ic-i_global[0]];	
		}

		bcastf_mpi(&corcon, 1, master_task);

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				cor[j][i] = corcon;	
			}
		}

		/*
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				f_cor[j][i] = cor[j][i];	
			}
		}
		*/
	}

	/*
	*f_rfe = rfe;
	*f_rfw = rfw;
	*f_rfn = rfn;
	*f_rfs = rfs;
	*/
	return;
}

//!=============================================================
//! Check if a  processor has ( i_in, j_in ) in its local domain.
//!-------------------------------------------------------------
int judge_inout(int i_in, int j_in,
				int imin_in, int imax_in,
				int jmin_in, int jmax_in){
	if ((i_in >= imin_in && i_in <= imax_in) &&
		(j_in >= jmin_in && j_in <= jmax_in)){
		return 1;	
	}else{
		return 0;	
	}
	return 0;

}

void read_trajdrf(){
	FILE *fp;
	int i;

	if ((fp = fopen("drf.list", "r")) == NULL){
		if (my_task == master_task)	
			printf("Read drf.list Error!\n");
	}

	fscanf(fp, "%d", &np);

	for (i = 0; i < np; i++){
		fscanf(fp, "%f", xstart+i);
		fscanf(fp, "%f", ystart+i);
	}

	fclose(fp);
}

/*
void bfrz_(int *f_mw, int *f_me, int *f_ms, int *f_mn,
		  int *f_nw, int *f_ne, int *f_ns, int *f_nn,
		  int *f_im, int *f_jm, int *f_nu, float f_frz[][i_size],
		  float *f_rdisp){
*/
void bfrz(int mw, int me, int ms, int mn,
		  int nw, int ne, int ns, int nn,
		  int im, int jm, int nu, float frz[][i_size]){

//!----------------------------------------------------------------------!
//!     calculate boundary flow-relaxation array "frz"                   !
//!----------------------------------------------------------------------!
//!     nu          = unit# (fort.nu) for ASCII printout                 !
//!     mw,me,ms,mn = #buffer grid-pnts near west, east, south & north   !
//!                   boundaries where assimilation frz=1.0; recommended !
//!                   buffer 100~200km;  m?=0 for no buffer;             !
//!                   As a precaution, program stops if 0<m?<=3 (or <0)  !
//!     nw,ne,ns,nn = n_west,n_east,n_south,n_north                      !
//!                   is =-1 if "bfrz" is being called by processor that !
//!                   shares the west, east, south or north boundary     !
//!                   Just set all to -1 for a single-processor run      !
//!     frz         = 1 at boundaries and tapered (tanh) =0  interior    !
//!                                                                      !
//!                ... l.oey --- lyo@princeton.edu (Jan/2008)            !
//!----------------------------------------------------------------------!


	/*
	int mw = *f_mw; 
	int me = *f_me; 
	int ms = *f_ms; 
	int mn = *f_mn; 
	int nw = *f_nw; 
	int ne = *f_ne; 
	int ns = *f_ns; 
	int nn = *f_nn; 

	int im = *f_im; 
	int jm = *f_jm; 
	int nu = *f_nu;

	float rdisp = *f_rdisp;
	*/

	int mmid;
	int i, j, ii, jj;
	float c, tanhm;

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			frz[j][i] = 0;//! Initialize interior	
		}
	}
//! West:
	if (nw == -1){
		if (mw > 3){//! west buffer: needs at least 4 pts
			for (j = 0; j < jm; j++){
				frz[j][0] = 1.0f;
			}
			mmid = mw/2;
			c = 5.0f/(float)(mw-mmid);

			for (i = 1; i < mw; i++){
				ii = i;
				tanhm = 0.5f*(1.0f-tanhf((float)(i+1-mmid)*c));
				for (j = 0; j < jm; j++){
					frz[j][ii] = tanhm;	
				}
			}
		}else if (mw == 0 || mw == 1){
			//! do nothing, i.e. frz remains = 0 for mw=0 or 1
		}else{
			//!mw=2 or 3 or <0
			printf("Stopped in bfrz. Proc = %d, mw = %d\n", my_task, mw);
			exit(1);
		}
	}

//! East:

	if (ne == -1){
		if (me > 3){//! east buffer:
			for (j = 0; j < jm; j++){
				frz[j][im-1] = 1.0f;	
			}
			mmid = me/2;
			c = 5.0f/(float)(me-mmid);

			for (i = 1; i < me; i++){
				ii = im-i-1;	
				tanhm = 0.5f*(1.0f - tanhf((float)(i+1-mmid)*c));
				for (j = 0; j < jm; j++){
					frz[j][ii] = tanhm;	
				}
			}
		}else if (me == 0 || me == 1){
			//! do nothing, i.e. frz remains = 0 for me=0 or 1
		}else{
			//!me=2 or 3 or <0
			printf("Stopped in bfrz. Proc %d, me = %d\n", my_task, me);	
			exit(1);
		}
	}

//! South:

	if (ns == -1){
		if (ms > 3){
			//! south buffer:
			for (i = 0; i < im; i++){
				frz[0][i] = 1.0f;
			}
			mmid = ms/2;
			c = 5.0f/(float)(ms-mmid);

			for (j = 1; j < ms; j++){
				jj = j;	
				tanhm = 0.5f*(1.0f-tanhf((float)(j+1-mmid)*c));
				for (i = 0; i < im; i++){
				//! lyo:debug:lyo:20110224:fxu:stcc:delete "if (nw.eq.-1.."
					frz[jj][i] = MAX(tanhm, frz[jj][i]);
						
				}
			}
		}else if (ms == 0 || ms == 1){
			//! do nothing, i.e. frz remains = 0 for ms=0 or 1
		}else{
			printf("Stopped in bfrz. Proc %d, ms = %d\n", my_task, ms);	
			exit(1);
		}
	}

//! North:
	
	if (nn == -1){
		if (mn > 3){//! north buffer:
			for (i = 0; i < im; i++){
				frz[jm-1][i] = 1.0f;	
			}
			mmid = mn/2;
			c = 5.0f/(float)(mn-mmid);
			for (j = 1; j < mn; j++){
				jj = jm-j-1;	
				tanhm = 0.5f*(1.0f-tanhf((float)(j+1-mmid)*c));
				for (i = 0; i < im; i++){
				//! lyo:debug:lyo:20110224:fxu:stcc:delete "if (nw.eq.-1.."
					frz[jj][i] = MAX(tanhm, frz[jj][i]);		
				}
			}
		}else if (mn == 0 || mn == 1){
			//! do nothing, i.e. frz remains = 0 for mn=0 or 1
		}else{
			printf("Stopped in bfrz. Proc %d, mn = %d\n", my_task, mn);	
			exit(1);
		}
	}

	/*	
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			frz[j][i] *= rdisp;
		}
	}


	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_frz[j][i] = frz[j][i];	
		}
	}
	*/

	return;
}

//void read_tide_(){
void read_tide(){
//! here is one example to read tidal amplitude & phase at the eastern 
//! boundary - intended for western N-Atlantic (profs) domain

	read_tide_east_pnetcdf(ampe, phae, amue, phue);
}

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
					 float f_v[][j_size][i_size]){
*/

void update_initial(){

//! update the initial conditions and set the remaining initial conditions
	int i, j, k;
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			ua[j][i] = uab[j][i];
			va[j][i] = vab[j][i];
			el[j][i] = elb[j][i];
			et[j][i] = etb[j][i];
			etf[j][i] = et[j][i];
			d[j][i] = h[j][i]+el[j][i];
			dt[j][i] = h[j][i]+et[j][i];
			w[0][j][i] = vfluxf[j][i];
		}	
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				l[k][j][i] = 0.1f*dt[j][i];
				q2b[k][j][i] = small;
				q2lb[k][j][i] = l[k][j][i]*q2b[k][j][i];
				kh[k][j][i] = l[k][j][i]*sqrtf(q2b[k][j][i]);
				km[k][j][i] = kh[k][j][i];
				kq[k][j][i] = kh[k][j][i];
				aam[k][j][i] = aam_init;
			}
		}
	}
	
	for (k = 0; k < kbm1; k++){
		for (i = 0; i < im; i++){
			for (j =0; j < jm; j++){
				q2[k][j][i] = q2b[k][j][i];
				q2l[k][j][i] = q2lb[k][j][i];
				t[k][j][i] = tb[k][j][i];
				s[k][j][i] = sb[k][j][i];
				u[k][j][i] = ub[k][j][i];
				v[k][j][i] = vb[k][j][i];
			}
		}
	}

	if (npg == 1){// !fhx:Toni:npg
		//baropg_(rho, drhox, dt, drhoy, ramp);
		//in fact drhox & drhoy are not assigned
		//for ramp is not initialized now, and is 0;
		baropg(rho, rmean, dum, dvm, dt, drhox, drhoy, ramp);
	}
	else if (npg == 2){
		baropg_mcc(rho, rmean, d, dum, dvm, dt, drhox, drhoy, ramp);
	}
	else{
		error_status = 1;
		printf("Error: invalid value for npg\n");
	}

	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				drx2d[j][i] = drx2d[j][i]+drhox[k][j][i]*dz[k];
				dry2d[j][i] = dry2d[j][i]+drhoy[k][j][i]*dz[k];
			}
		}
	}
	

/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_ua[j][i] = ua[j][i];
			f_va[j][i] = va[j][i];
			f_el[j][i] = el[j][i];
			f_et[j][i] = et[j][i];
			f_etf[j][i] = etf[j][i];
			f_d[j][i] = d[j][i];
			f_dt[j][i] = dt[j][i];
			f_w[0][j][i] = w[0][j][i];

			f_drx2d[j][i] = drx2d[j][i];
			f_dry2d[j][i] = dry2d[j][i];
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				f_l[k][j][i] = l[k][j][i];
				f_q2b[k][j][i] = q2b[k][j][i];
				f_q2lb[k][j][i] = q2lb[k][j][i];
				f_kh[k][j][i] = kh[k][j][i];
				f_km[k][j][i] = km[k][j][i];
				f_kq[k][j][i] = kq[k][j][i];
				f_aam[k][j][i] = aam[k][j][i];

				f_drhox[k][j][i] = drhox[k][j][i];
				f_drhoy[k][j][i] = drhoy[k][j][i];
			}
		}
	}

	for (k = 0; k < kbm1; k++){
		for (i = 0; i < im; i++){
			for (j = 0; j < jm; j++){
				f_q2[k][j][i] = q2[k][j][i];	
				f_q2l[k][j][i] = q2l[k][j][i];
				f_t[k][j][i] = t[k][j][i];
				f_s[k][j][i] = s[k][j][i];
				f_u[k][j][i] = u[k][j][i];
				f_v[k][j][i] = v[k][j][i];
			}
		}
	}
*/
	return;
}


/*
void bottom_friction_(float f_cbc[][i_size],
					  float f_aamfrz[][i_size]){
*/
void bottom_friction(){

//! calculate the bottom friction coefficient

	int i, j;

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			aamfrz[j][i] = f_aamfrz[j][i];	
		}
	}
	*/

//! calculate bottom friction
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			float tmp = (kappa/logf((1.0f+zz[kbm1-1])*h[j][i]/z0b));
			cbc[j][i] = tmp*tmp;
			cbc[j][i] = MAX(cbcmin, cbc[j][i]);
//! if the following is invoked, then it is probable that the wrong
//! choice of z0b or vertical spacing has been made:
			cbc[j][i] = MIN(cbcmax, cbc[j][i])*(1.0f+aamfrz[j][i]);
			
			//f_cbc[j][i] = cbc[j][i];
		}
	}
	return;
}

/*
void incmix_(float f_aam[][j_size][i_size], 
			 int *f_im, int *f_jm, int *f_kb,
			 float x[][i_size], float y[][i_size]){
*/
void incmix(float aam[][j_size][i_size], 
			int im, int jm, int kb,
			float x[][i_size], float y[][i_size]){
			 

//!     Increase 'aam' by '(1+fac)*' at (lono,lato), then taper off to '1'
//!       in gaussian-manner for (x-xo, y-yo) > xs and ys
//!
//!     Inputs: aam,x,y,xs,ys & fac
//!     Output: aam is modified

	/*
	int im = *f_im;
	int jm = *f_jm;
	int kb = *f_kb;
	*/
	int i, j, k;
	float factor, expon;

/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				aam[k][j][i] = f_aam[k][j][i];
			}
		}
	}
*/

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				factor = 0;
				float tmp_x = (x[j][i]-lono)/xs;
				float tmp_y = (y[j][i]-lato)/ys;
				expon = (tmp_x*tmp_x)+(tmp_y*tmp_y);
				if (expon <= 10.0f){
					factor = fak*expf(-expon);	
				}
				aam[k][j][i] = aam[k][j][i]*(1.0f+factor);
			}
		}
	}

/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				f_aam[k][j][i] = aam[k][j][i];
			}
		}
	}
*/

	return;
}
