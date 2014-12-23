#include<stdio.h>
#include<math.h>
#include"cadvance.h"
#include"cadvance_gpu.h"
#include"data.h"
#include"csolver.h"
#include"timer_all.h"
#include"cparallel_mpi.h"


//void advance_c_(int *f_iint){
void advance(){

	//get_time();
	get_time_gpu();

	//surface_forcing();
	surface_forcing_gpu();

	//momentum3d();
	momentum3d_gpu();

	//mode_interaction();
	mode_interaction_gpu();

	//printf("iint = %d\n", iint);

	for (iext = 1; iext <= isplit; iext++){
		//mode_external();	
		mode_external_gpu();	
	}

	//mode_internal();
	mode_internal_gpu();
//	mode_internal();

	print_section();

//	check_nan();
//
	//store_mean();
	store_mean_gpu();
//
	//store_surf_mean();
	store_surf_mean_gpu();
//
//	write_output();
//
//	if (iint % irestart == 0)
//		write_restart_pnetcdf();
	
	//check_velocity();
	check_velocity_gpu();
}



/*
void get_time_c_(float *f_time, 
               float *f_ramp,
			   int *f_iint){
*/

void get_time(){
//void get_time(){
	//iint = *f_iint;
	
	model_time = dti*((float)iint)/86400.f+time0;
	ramp = 1.0f;	

	/*
	if (lramp){
		ramp = time/period;
		if (ramp > 1.0f)
			ramp = 1.0f;
	}else{
		ramp = 1.0f;	
	}
	*/

	/*
	*/

	/*
	*f_time = model_time;
	*f_ramp = ramp;
	*/

	return;
}

/*
void surface_forcing_c_(float f_vfluxf[][i_size], 
					  float f_t[][j_size][i_size],
					  float f_e_atmos[][i_size],
					  float f_w[][j_size][i_size],
					  float f_swrad[][i_size]){
*/

void surface_forcing(){

//! set time dependent surface boundary conditions
//!     BUT, wusurf, wvsurf, vfluxf, wtsurf and wssurf are calc. in different 
//!     subroutines as indicated below. To specify idealized forcing for 
//!     these quantities, set calc_wind, calc_river and calc_tsforce to 
//!     "false" then specify here
					  
	int i,j;
    float tatm,satm;
	
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){

			
			/*
			vfluxf[j][i] = f_vfluxf[j][i];
			t[0][j][i] = f_t[0][j][i];
			*/
			

			e_atmos[j][i] = 0.0f;
//! wind stress
//! value is negative for westerly or southerly winds. The wind stress
//! should be tapered along the boundary to suppress numerically induced
//! oscilations near the boundary (Jamart and Ozer, JGR, 91, 10621-10631)
//
//! wusurf, wvsurf & vfluxf assume values (usuaully=0) from initialize.f
//! ususually do NOT set here, as they are calculated in wind.f & river.f
//			wusurf[j][i] = 0.0f;		!calculated in wind.f
//			wvsurf[j][i] = 0.0f;		!calculated in wind.f
//			vfluxf[j][i] = 0.0f;		!calculated in river.f

//! set w(i,j,1)=vflux(i,j).ne.0 if one wishes non-zero flow across
//! the sea surface. See calculation of elf(i,j) below and subroutines
//! vertvl, advt1 (or advt2). If w(1,j,1)=0, and, additionally, there
//! is no net flow across lateral boundaries, the basin volume will be
//! constant; if also vflux(i,j).ne.0, then, for example, the average
//! salinity will change and, unrealistically, so will total salt
			w[0][j][i] = vfluxf[j][i];

//! set wtsurf to the sensible heat, the latent heat (which involves
//! only the evaporative component of vflux) and the long wave
//! radiation
//! wtsurf & wssurf are calculated in tsclim_monthly //			wtsurf[j][i] = 0.0f;

//! set swrad to the short wave radiation
			swrad[j][i] = 0.0f;

//! to account for change in temperature of flow crossing the sea
//! surface (generally quite small compared to latent heat effect)
			//tatm = t[0][j][i]+tbias;	//! an approximation
			
//			wtsurf[j][i] = wtsurf[j][i]+vfluxf[j][i]*(tatm-t[0][j][i]-tbias);

//! set the salinity of water vapor/precipitation which enters/leaves
//! the atmosphere (or e.g., an ice cover)
			//satm = 0.0f;

//			wssurf[j][i] = vfluxf[j][i]*(satm-s[0][j][i]-sbias);

			
			/*
			f_e_atmos[j][i] = e_atmos[j][i];
			f_w[0][j][i] = w[0][j][i];
			f_swrad[j][i] = swrad[j][i];
			*/

		}
	}
    return;
}

/*
void momentum3d_c_(float f_advx[][j_size][i_size],
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
				float f_d[][i_size]){	//!lyo:!stokes:change subr name
*/

void momentum3d(){
//! formerly subroutine lateral_viscosity
//! calculate horizontal 3d-Momentum terms including the lateral viscosity
	int i, j, k;
	if (mode != 2){
//------------------------------------------------------------------------
		//advct(a, c, ee);
		/*
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					u[k][j][i] = f_u[k][j][i];
					v[k][j][i] = f_v[k][j][i];
					ub[k][j][i] = f_ub[k][j][i];
					vb[k][j][i] = f_vb[k][j][i];
					aam[k][j][i] = f_aam[k][j][i];
					//advx[k][j][i] = f_advx[k][j][i];
					//advy[k][j][i] = f_advy[k][j][i];
				}
			}
		}

		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					rho[k][j][i] = f_rho[k][j][i];
					rmean[k][j][i] = f_rmean[k][j][i];
				}
			}
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				dum[j][i] = f_dum[j][i];
				dvm[j][i] = f_dvm[j][i];
				dt[j][i] = f_dt[j][i];
				d[j][i] = f_d[j][i];
			}
		}
		*/

		
		advct(advx,v,u,dt,ub,aam,vb,advy);


//------------------------------------------------------------------------
		if (calc_stokes){
			printf("calc_stokes = TRUE!\n");
			printf("But stokes function is not implemented now...\n");
			exit(1);
			//stokes(ee);		//!(a,c,ee) !lyo:!stokes:
		}
//------------------------------------------------------------------------




		if (npg == 1){
			//baropg();	
			baropg(rho, rmean, dum, dvm, dt, drhox, drhoy, ramp);
			//baropg_c_(rho, rmean, dum, dvm, dt, drhox, drhoy, &ramp);
		}else if (npg == 2){
			//baropg_mcc();	
			baropg_mcc(rho, rmean, d, dum, dvm, dt, drhox, drhoy, ramp);	

		}else{
			error_status = 1;	
			printf("Error: invalid value for npg, File:%s, Func:%s, Line:%d",
					__FILE__, __func__, __LINE__);
		}

//------------------------------------------------------------------------

//! if mode=2 then initial values of aam2d are used. If one wishes
//! to use Smagorinsky lateral viscosity and diffusion for an
//! external (2-D) mode calculation, then appropiate code can be
//! adapted from that below and installed just before the end of the
//! "if(mode.eq.2)" loop in subroutine advave
//
//! calculate Smagorinsky lateral viscosity:
//! ( hor visc = horcon*dx*dy*sqrt((du/dx)**2+(dv/dy)**2
//!                                +.5*(du/dy+dv/dx)**2) )
//!lyo:scs1d:
		
		if (n1d != 0){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						aam[k][j][i] = aam_init;	
					}
				}
			}
		}else{
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					for (i = 1; i < imm1; i++){
						float tmp_u = (u[k][j][i+1]-u[k][j][i])/dx[j][i];
						float tmp_v = (v[k][j+1][i]-v[k][j][i])/dy[j][i];
						float tmp_uv = (0.25f*(u[k][j+1][i]
											  +u[k][j+1][i+1]
											  -u[k][j-1][i]
											  -u[k][j-1][i+1])/dy[j][i]
									   +0.25f*(v[k][j][i+1]
									   	      +v[k][j+1][i+1]
									   	      -v[k][j][i-1]
									   	      -v[k][j+1][i-1])/dx[j][i]);

						aam[k][j][i]=horcon*dx[j][i]*dy[j][i]
									*(1.f+aamfrz[j][i])		//!lyo:channel:
									*sqrtf((tmp_u*tmp_u)
										  +(tmp_v*tmp_v)
										  +0.5f*(tmp_uv*tmp_uv));
					}
				}
			}
		}

		exchange3d_mpi(aam, im, jm, kbm1);

		/*
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					f_advx[k][j][i] = advx[k][j][i];
					f_advy[k][j][i] = advy[k][j][i];
					f_drhox[k][j][i] = drhox[k][j][i];
					f_drhoy[k][j][i] = drhoy[k][j][i];

					f_rho[k][j][i] = rho[k][j][i];
					f_aam[k][j][i] = aam[k][j][i];
				}
			}
		}
		*/

	}

	return;
}


/*
void mode_interaction_c_(float f_adx2d[][i_size], float f_ady2d[][i_size],
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
					  float f_el[][i_size], float f_aamfrz[][i_size]){
*/

void mode_interaction(){
	struct timeval time_start_mode_interaction, 
				   time_end_mode_interaction;
	int i,j,k;

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				advx[k][j][i] = f_advx[k][j][i];
				advy[k][j][i] = f_advy[k][j][i];
				drhox[k][j][i] = f_drhox[k][j][i];
				drhoy[k][j][i] = f_drhoy[k][j][i];
				aam[k][j][i] = f_aam[k][j][i];
			}
		}
	}
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			d[j][i] = f_d[j][i];
			ua[j][i] = f_ua[j][i];
			va[j][i] = f_va[j][i];
			uab[j][i] = f_uab[j][i];
			vab[j][i] = f_vab[j][i];
			el[j][i] = f_el[j][i];
			wubot[j][i] = f_wubot[j][i];
			wvbot[j][i] = f_wvbot[j][i];
			aamfrz[j][i] = f_aamfrz[j][i];

			adx2d[j][i] = f_adx2d[j][i];
			ady2d[j][i] = f_ady2d[j][i];
			drx2d[j][i] = f_drx2d[j][i];
			dry2d[j][i] = f_dry2d[j][i];
			aam2d[j][i] = f_aam2d[j][i];
			advua[j][i] = f_advua[j][i];
			advva[j][i] = f_advva[j][i];
		}
	}
	*/


	if(mode != 2){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				adx2d[j][i] = 0.0f;
				ady2d[j][i] = 0.0f;
				drx2d[j][i] = 0.0f;
				dry2d[j][i] = 0.0f;
				aam2d[j][i] = 0.0f;
			}
		}
	
		for(k = 0; k < kbm1; k++){
			for(j = 0; j < jm; j++){
				for(i = 0; i < im; i++){
					adx2d[j][i] = adx2d[j][i]+advx[k][j][i]*dz[k];
					ady2d[j][i] = ady2d[j][i]+advy[k][j][i]*dz[k];
					drx2d[j][i] = drx2d[j][i]+drhox[k][j][i]*dz[k];
					dry2d[j][i] = dry2d[j][i]+drhoy[k][j][i]*dz[k];
					aam2d[j][i] = aam2d[j][i]+aam[k][j][i]*dz[k];
				}
			}
		}

		/*
        advave(advua,d,ua,va,uab,aam2d,
				vab,advva,wubot,wvbot);
		*/

		advave();

		for(j = 0;  j < jm; j++){
			for(i = 0; i < im; i++){
				adx2d[j][i] = adx2d[j][i]-advua[j][i];
				ady2d[j][i] = ady2d[j][i]-advva[j][i]; 
			}
		}
	}

	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			egf[j][i] = el[j][i]*ispi;	
		}
	}

	for(j = 0; j < jm; j++){
		for(i = 1; i < im; i++){
			utf[j][i] = ua[j][i]*(d[j][i]+d[j][i-1])*isp2i;
		}
	}

	for(j = 1; j < jm; j++){
		for(i = 0; i < im; i++){
			vtf[j][i] = va[j][i]*(d[j][i]+d[j-1][i])*isp2i;
		}	
	}
      
	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_adx2d[j][i] = adx2d[j][i];
			f_ady2d[j][i] = ady2d[j][i];
			f_drx2d[j][i] = drx2d[j][i];
			f_dry2d[j][i] = dry2d[j][i];

			f_aam2d[j][i] = aam2d[j][i];

			f_advua[j][i] = advua[j][i];
			f_advva[j][i] = advva[j][i];

			f_wubot[j][i] = wubot[j][i];
			f_wvbot[j][i] = wvbot[j][i];

			f_egf[j][i] = egf[j][i];
			f_utf[j][i] = utf[j][i];
			f_vtf[j][i] = vtf[j][i];
		}
	}
	*/

	return;
	
}

/*
void mode_external_c_(float f_elf[][i_size], float f_advua[][i_size],
				   float f_advva[][i_size], float f_fluxua[][i_size],
				   float f_fluxva[][i_size], float f_uaf[][i_size],
				   float f_vaf[][i_size], float f_etf[][i_size],
				   float f_ua[][i_size], float f_va[][i_size],
				   float f_el[][i_size], float f_elb[][i_size],
				   float f_d[][i_size], float f_uab[][i_size],
				   float f_vab[][i_size], float f_egf[][i_size],
				   float f_utf[][i_size], float f_vtf[][i_size],
				
				   float f_vfluxf[][i_size], float f_wusurf[][i_size],
				   float f_aam2d[][i_size], float f_wubot[][i_size],
				   float f_wvbot[][i_size], float f_adx2d[][i_size],
				   float f_drx2d[][i_size], float f_ady2d[][i_size],
				   float f_wvsurf[][i_size], 
				   float f_dry2d[][i_size], int *f_iext,
				   float f_e_atmos[][i_size]){
*/

//void mode_external_(int *f_iext){

void mode_external(){
	struct timeval time_start_mode_external, 
				   time_end_mode_external;
				   
	int i,j;
	float fluxua[j_size][i_size];
	float fluxva[j_size][i_size];

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			d[j][i] = f_d[j][i];	
			elb[j][i] = f_elb[j][i];
			ua[j][i] = f_ua[j][i];
			va[j][i] = f_va[j][i];
			uab[j][i] = f_uab[j][i];
			aam2d[j][i] = f_aam2d[j][i];
			vab[j][i] = f_vab[j][i];
			adx2d[j][i] = f_adx2d[j][i];
			el[j][i] = f_el[j][i];
			drx2d[j][i] = f_drx2d[j][i];
			wusurf[j][i] = f_wusurf[j][i];
			wubot[j][i] = f_wubot[j][i];
			ady2d[j][i] = f_ady2d[j][i];
			dry2d[j][i] = f_dry2d[j][i];
			wvsurf[j][i] = f_wvsurf[j][i];
			wvbot[j][i] = f_wvbot[j][i];
			etf[j][i] = f_etf[j][i];
			egf[j][i] = f_egf[j][i];
			vtf[j][i] = f_vtf[j][i];
			utf[j][i] = f_utf[j][i];
			vfluxf[j][i] = f_vfluxf[j][i];
			e_atmos[j][i] = f_e_atmos[j][i];
		}
	}
	*/

	//iext = *f_iext;
	
/*
      do j=2,jm
        do i=2,im
          fluxua(i,j)=.25e0*(d(i,j)+d(i-1,j))
     $                 *(dy(i,j)+dy(i-1,j))*ua(i,j)
          fluxva(i,j)=.25e0*(d(i,j)+d(i,j-1))
     $                 *(dx(i,j)+dx(i,j-1))*va(i,j)
        end do
      end do
*/
	for(j = 1; j < jm; j++){
		for(i = 1; i < im; i++){
			fluxua[j][i] = 0.25f*(d[j][i]+d[j][i-1])
				*(dy[j][i]+dy[j][i-1])*ua[j][i];
			fluxva[j][i] = 0.25f*(d[j][i]+d[j-1][i])
				*(dx[j][i]+dx[j-1][i])*va[j][i];
		}
	}
/*
! NOTE addition of surface freshwater flux, w(i,j,1)=vflux, compared
! with pom98.f. See also modifications to subroutine vertvl
      do j=2,jmm1
        do i=2,imm1
          elf(i,j)=elb(i,j)
     $              +dte2*(-(fluxua(i+1,j)-fluxua(i,j)
     $                      +fluxva(i,j+1)-fluxva(i,j))/art(i,j)
     $                      -vfluxf(i,j))
        end do
      end do
*/
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			elf[j][i] = elb[j][i]
						+dte2*(-(fluxua[j][i+1]-fluxua[j][i]
									+fluxva[j+1][i]-fluxva[j][i])/art[j][i]
								-vfluxf[j][i]);
		}
	}

      //call bcond(1)
#ifndef TIME_DISABLE
	timer_now(&time_start_mode_external);
#endif

	//bcond_xsz_(1);//modify elf boundary //just refernce fsm
	bcond(1);//modify elf boundary //just refernce fsm

#ifndef TIME_DISABLE
	timer_now(&time_end_mode_external);
	bcond_time += time_consumed(&time_start_mode_external,
								&time_end_mode_external);

#endif
	//call exchange2d_mpi(elf,im,jm)
	//exchange2d_mpi_xsz_(elf,im,jm);
	exchange2d_mpi(elf,im,jm);

	//if(mod(iext,ispadv).eq.0) call advave(advua,d,ua,va,
	//$      fluxua,fluxva,uab,aam2d,vab,advva,wubot,wvbot)
	
	if ((iext % ispadv) == 0){
#ifndef TIME_DISABLE
		timer_now(&time_start_mode_external);
#endif

		//advave_(advua,d,ua,va,fluxua,fluxva,uab,aam2d,vab,advva,wubot,wvbot);
		advave();

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_external);
		advave_time += time_consumed(&time_start_mode_external,
									 &time_end_mode_external);
#endif
	}

/*
      do j=2,jmm1
        do i=2,im
          uaf(i,j)=adx2d(i,j)+advua(i,j)
     $              -aru(i,j)*.25e0
     $                *(cor(i,j)*d(i,j)*(va(i,j+1)+va(i,j))
     $                 +cor(i-1,j)*d(i-1,j)*(va(i-1,j+1)+va(i-1,j)))
     $              +.25e0*grav*(dy(i,j)+dy(i-1,j))
     $                *(d(i,j)+d(i-1,j))
     $                *((1.e0-2.e0*alpha)
     $                   *(el(i,j)-el(i-1,j))
     $                  +alpha*(elb(i,j)-elb(i-1,j)
     $                         +elf(i,j)-elf(i-1,j))
     $                  +e_atmos(i,j)-e_atmos(i-1,j))
     $              +drx2d(i,j)+aru(i,j)*(wusurf(i,j)-wubot(i,j))
        end do
      end do
*/

	for(j = 1; j < jmm1; j++){
		for(i = 1; i < im; i++){
			uaf[j][i] = adx2d[j][i]+advua[j][i]-aru[j][i]*0.25f
							*(cor[j][i]*d[j][i]*(va[j+1][i]+va[j][i])
								+cor[j][i-1]*d[j][i-1]*(va[j+1][i-1]+va[j][i-1]))
						+0.25f*grav*(dy[j][i]+dy[j][i-1])
							*(d[j][i]+d[j][i-1])
							*((1.0f-2.0f*alpha)*(el[j][i]-el[j][i-1])
								+alpha*(elb[j][i]-elb[j][i-1]
									+elf[j][i]-elf[j][i-1])
								+e_atmos[j][i]-e_atmos[j][i-1])
						+drx2d[j][i]+aru[j][i]*(wusurf[j][i]-wubot[j][i]);
		}
	}

/*
      do j=2,jmm1
        do i=2,im
          uaf(i,j)=((h(i,j)+elb(i,j)+h(i-1,j)+elb(i-1,j))
     $                *aru(i,j)*uab(i,j)
     $              -4.e0*dte*uaf(i,j))
     $             /((h(i,j)+elf(i,j)+h(i-1,j)+elf(i-1,j))
     $                 *aru(i,j))
        end do
      end do
*/

	for(j = 1; j < jmm1; j++){
		for(i = 1; i < im; i++){
			uaf[j][i] = ((h[j][i]+elb[j][i]+h[j][i-1]+elb[j][i-1])
								*aru[j][i]*uab[j][i]
							-4.0f*dte*uaf[j][i])
						/((h[j][i]+elf[j][i]+h[j][i-1]+elf[j][i-1])
							*aru[j][i]);
		}
	}

/*
      do j=2,jm
        do i=2,imm1
          vaf(i,j)=ady2d(i,j)+advva(i,j)
     $              +arv(i,j)*.25e0
     $                *(cor(i,j)*d(i,j)*(ua(i+1,j)+ua(i,j))
     $               +cor(i,j-1)*d(i,j-1)*(ua(i+1,j-1)+ua(i,j-1)))
     $              +.25e0*grav*(dx(i,j)+dx(i,j-1))
     $                *(d(i,j)+d(i,j-1))
     $                *((1.e0-2.e0*alpha)*(el(i,j)-el(i,j-1))
     $                  +alpha*(elb(i,j)-elb(i,j-1)
     $                         +elf(i,j)-elf(i,j-1))
     $                  +e_atmos(i,j)-e_atmos(i,j-1))
     $              +dry2d(i,j)+arv(i,j)*(wvsurf(i,j)-wvbot(i,j))
        end do
      end do
*/

	for(j = 1; j < jm; j++){
		for(i = 1; i < imm1; i++){
			vaf[j][i] = ady2d[j][i]+advva[j][i]
							+arv[j][i]*0.25f
								*(cor[j][i]*d[j][i]*(ua[j][i+1]+ua[j][i])
									+cor[j-1][i]*d[j-1][i]*(ua[j-1][i+1]+ua[j-1][i]))
							+0.25f*grav*(dx[j][i]+dx[j-1][i])
								*(d[j][i]+d[j-1][i])
								*((1.0f-2.0f*alpha)*(el[j][i]-el[j-1][i])
									+alpha*(elb[j][i]-elb[j-1][i]
										+elf[j][i]-elf[j-1][i])
									+e_atmos[j][i]-e_atmos[j-1][i])
							+dry2d[j][i]+arv[j][i]*(wvsurf[j][i]-wvbot[j][i]);
		}
	}

/*
      do j=2,jm
        do i=2,imm1
          vaf(i,j)=((h(i,j)+elb(i,j)+h(i,j-1)+elb(i,j-1))
     $                *vab(i,j)*arv(i,j)
     $              -4.e0*dte*vaf(i,j))
     $             /((h(i,j)+elf(i,j)+h(i,j-1)+elf(i,j-1))
     $                 *arv(i,j))
        end do
      end do
*/
	for(j = 1; j < jm; j++){
		for(i = 1; i < imm1; i++){
			vaf[j][i] = ((h[j][i]+elb[j][i]+h[j-1][i]+elb[j-1][i])
								*vab[j][i]*arv[j][i]
							-4.0f*dte*vaf[j][i])
						/((h[j][i]+elf[j][i]+h[j-1][i]+elf[j-1][i])
							*arv[j][i]);
		}
	}

    //call bcond(2)
#ifndef TIME_DISABLE
	timer_now(&time_start_mode_external);
#endif

    bcond(2);//modify uaf&vaf boundary condition

#ifndef TIME_DISABLE
	timer_now(&time_end_mode_external);
	bcond_time += time_consumed(&time_start_mode_external,
								&time_end_mode_external);
#endif

    //call exchange2d_mpi(uaf,im,jm)
    //call exchange2d_mpi(vaf,im,jm)

    //exchange2d_mpi_xsz_(uaf,im,jm);
    exchange2d_mpi(uaf,im,jm);
    //exchange2d_mpi_xsz_(vaf,im,jm);
    exchange2d_mpi(vaf,im,jm);

/*
      if(iext.eq.(isplit-2))then
        do j=1,jm
          do i=1,im
            etf(i,j)=.25e0*smoth*elf(i,j)
          end do
        end do

      else if(iext.eq.(isplit-1)) then

        do j=1,jm
          do i=1,im
            etf(i,j)=etf(i,j)+.5e0*(1.-.5e0*smoth)*elf(i,j)
          end do
        end do

      else if(iext.eq.isplit) then

        do j=1,jm
          do i=1,im
            etf(i,j)=(etf(i,j)+.5e0*elf(i,j))*fsm(i,j)
          end do
        end do

      end if
*/
	if(iext == (isplit-2)){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				etf[j][i] = 0.25f*smoth*elf[j][i];
			}
		}
	}
    else if(iext == (isplit-1)){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				etf[j][i] = etf[j][i]+0.5f*(1.0f-0.5f*smoth)*elf[j][i];
			}
		}
    }
    else if(iext == isplit){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				etf[j][i] = (etf[j][i]+0.5f*elf[j][i])*fsm[j][i];
			}
		}
    }

/*
!lyo:!vort:beg:Vorticity(e.g. JEBAR) analysis once only at iext=isplit
!     if (iint.eq.1 .or. mod(iint,iprints).eq.0) then
      if ( calc_vort )
     $   CALL VORT !(nhra,nu) !(ADVUA,ADVVA,ADX2D,ADY2D,DRX2D,DRY2D,IM,JM)
!     endif
!lyo:!vort:end:
      end if  !if(iext.eq.isplit) then...
*/

	if (calc_vort) vort();

//! apply filter to remove time split
/*
      do j=1,jm
        do i=1,im
          ua(i,j)=ua(i,j)+.5e0*smoth*(uab(i,j)-2.e0*ua(i,j)+uaf(i,j))
          va(i,j)=va(i,j)+.5e0*smoth*(vab(i,j)-2.e0*va(i,j)+vaf(i,j))
          el(i,j)=el(i,j)+.5e0*smoth*(elb(i,j)-2.e0*el(i,j)+elf(i,j))
          elb(i,j)=el(i,j)
          el(i,j)=elf(i,j)
          d(i,j)=h(i,j)+el(i,j)
          uab(i,j)=ua(i,j)
          ua(i,j)=uaf(i,j)
          vab(i,j)=va(i,j)
          va(i,j)=vaf(i,j)
        end do
      end do
*/

	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			ua[j][i] = ua[j][i]+0.5f*smoth*(uab[j][i]-2.0f*ua[j][i]+uaf[j][i]);
			va[j][i] = va[j][i]+0.5f*smoth*(vab[j][i]-2.0f*va[j][i]+vaf[j][i]);
			el[j][i] = el[j][i]+0.5f*smoth*(elb[j][i]-2.0f*el[j][i]+elf[j][i]);
			elb[j][i] = el[j][i];
			el[j][i] = elf[j][i];
			d[j][i] = h[j][i]+el[j][i];
			uab[j][i] = ua[j][i];
			ua[j][i] = uaf[j][i];
			vab[j][i] = va[j][i];
			va[j][i] = vaf[j][i];
		}
	}

/*
      if(iext.ne.isplit) then
        do j=1,jm
          do i=1,im
            egf(i,j)=egf(i,j)+el(i,j)*ispi
          end do
        end do
        do j=1,jm
          do i=2,im
            utf(i,j)=utf(i,j)+ua(i,j)*(d(i,j)+d(i-1,j))*isp2i
          end do
        end do
        do j=2,jm
          do i=1,im
            vtf(i,j)=vtf(i,j)+va(i,j)*(d(i,j)+d(i,j-1))*isp2i
          end do
        end do
       end if
*/

	if(iext != isplit){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				egf[j][i] = egf[j][i]+el[j][i]*ispi;
			}
		}
		for(j = 0; j < jm; j++){
			for(i = 1; i < im; i++){
				utf[j][i] = utf[j][i]+ua[j][i]*(d[j][i]+d[j][i-1])*isp2i;
			}
		}
		for(j = 1; j < jm; j++){
			for(i = 0; i < im; i++){
				vtf[j][i] = vtf[j][i]+va[j][i]*(d[j][i]+d[j-1][i])*isp2i;
			}
		}
	}

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_elf[j][i] = elf[j][i];
			f_advua[j][i] = advua[j][i];
			f_advva[j][i] = advva[j][i];
			f_wubot[j][i] = wubot[j][i];
			f_wvbot[j][i] = wvbot[j][i];
			f_aam2d[j][i] = aam2d[j][i];
			//f_fluxua[j][i] = fluxua[j][i];
			//f_fluxva[j][i] = fluxva[j][i];
			f_uaf[j][i] = uaf[j][i];
			f_vaf[j][i] = vaf[j][i];
			f_etf[j][i] = etf[j][i];
			f_ua[j][i] = ua[j][i];
			f_va[j][i] = va[j][i];
			f_el[j][i] = el[j][i];
			f_elb[j][i] = elb[j][i];
			f_d[j][i] = d[j][i];
			f_uab[j][i] = uab[j][i];
			f_vab[j][i] = vab[j][i];
			f_egf[j][i] = egf[j][i];
			f_utf[j][i] = utf[j][i];
			f_vtf[j][i] = vtf[j][i];
		}
	}
	*/
 /*
	call call_time_end(mode_external_time_end_xsz)
      mode_external_time_xsz = mode_external_time_end_xsz -
     $                         mode_external_time_start_xsz +
     $                         mode_external_time_xsz
*/
    return;
}


/*
void mode_internal_c_(float f_u[][j_size][i_size], float f_v[][j_size][i_size],
				   float f_w[][j_size][i_size], float f_uf[][j_size][i_size],
				   float f_vf[][j_size][i_size], 
				   float f_km[][j_size][i_size], float f_kh[][j_size][i_size],
				   float f_q2b[][j_size][i_size], float f_q2lb[][j_size][i_size],
				   float f_q2l[][j_size][i_size], float f_q2[][j_size][i_size],
				   float f_t[][j_size][i_size], float f_tb[][j_size][i_size],
				   float f_s[][j_size][i_size], float f_sb[][j_size][i_size],
				   float f_wubot[j_size][i_size], float f_wvbot[j_size][i_size],
				   float f_egb[][i_size], float f_etb[][i_size],
				   float f_et[][i_size], float f_dt[][i_size], 
				   float f_utb[][i_size], float f_vtb[][i_size],
				   float f_vfluxb[][i_size], float f_vfluxf[][i_size],

				   int *f_iint, float f_aam[][j_size][i_size],
				   float f_rho[][j_size][i_size], float f_advx[][j_size][i_size],
				   float f_advy[][j_size][i_size], float f_ub[][j_size][i_size],
				   float f_vb[][j_size][i_size], float f_utf[][i_size],
				   float f_vtf[][i_size], float f_etf[][i_size],
				   float f_wusurf[][i_size], float f_wvsurf[][i_size],
				   float f_egf[][i_size], float f_e_atmos[][i_size],
				   float f_drhox[][j_size][i_size], float f_drhoy[][j_size][i_size],
				   float f_wtsurf[][i_size], float f_wssurf[][i_size],
				   float f_swrad[][i_size]){
*/

//void mode_internal(int *f_iint){
void mode_internal(){
	struct timeval time_start_mode_internal, 
				   time_end_mode_internal;

	int i, j, k;
    float dxr,dxl,dyt,dyb;
	float tps[j_size][i_size];

	
	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				
				u[k][j][i] = f_u[k][j][i];	
				v[k][j][i] = f_v[k][j][i];
				q2b[k][j][i] = f_q2b[k][j][i];
				q2[k][j][i] = f_q2[k][j][i];
				aam[k][j][i] = f_aam[k][j][i];
				q2lb[k][j][i] = f_q2lb[k][j][i];
				q2l[k][j][i] = f_q2l[k][j][i];
				

			
				t[k][j][i] = f_t[k][j][i];
				s[k][j][i] = f_s[k][j][i];
				rho[k][j][i] = f_rho[k][j][i];
				tb[k][j][i] = f_tb[k][j][i];
				sb[k][j][i] = f_sb[k][j][i];

				
				advx[k][j][i] = f_advx[k][j][i];
				advy[k][j][i] = f_advy[k][j][i];

				ub[k][j][i] = f_ub[k][j][i];
				vb[k][j][i] = f_vb[k][j][i];
			
				drhox[k][j][i] = f_drhox[k][j][i];
				drhoy[k][j][i] = f_drhoy[k][j][i];


				//km[k][j][i] = f_km[k][j][i];
				//kh[k][j][i] = f_kh[k][j][i];
				


			}
		}
	}
	
	
	
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			utb[j][i] = f_utb[j][i];
			utf[j][i] = f_utf[j][i];
			dt[j][i] = f_dt[j][i];
			vtb[j][i] = f_vtb[j][i];
			vtf[j][i] = f_vtf[j][i];
			etf[j][i] = f_etf[j][i];
			etb[j][i] = f_etb[j][i];
			vfluxb[j][i] = f_vfluxb[j][i];
			vfluxf[j][i] = f_vfluxf[j][i];
			wusurf[j][i] = f_wusurf[j][i];
			wvsurf[j][i] = f_wvsurf[j][i];
			wubot[j][i] = f_wubot[j][i];
			wvbot[j][i] = f_wvbot[j][i];
			egf[j][i] = f_egf[j][i];
			egb[j][i] = f_egb[j][i];
			e_atmos[j][i] = f_e_atmos[j][i];
			wtsurf[j][i] = f_wtsurf[j][i];
			wssurf[j][i] = f_wssurf[j][i];
			swrad[j][i] = f_swrad[j][i];
			et[j][i] = f_et[j][i];
		
		}
	}
	*/
	
	
	//iint = *f_iint;

/*
      if((iint.ne.1.or.time0.ne.0.e0).and.mode.ne.2) then

! adjust u(z) and v(z) such that depth average of (u,v) = (ua,va)
        do j=1,jm
          do i=1,im
            tps(i,j)=0.e0
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              tps(i,j)=tps(i,j)+u(i,j,k)*dz(k)
            end do
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=2,im
              u(i,j,k)=(u(i,j,k)-tps(i,j))+
     $                 (utb(i,j)+utf(i,j))/(dt(i,j)+dt(i-1,j))
            end do
          end do
        end do

        do j=1,jm
          do i=1,im
            tps(i,j)=0.e0
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              tps(i,j)=tps(i,j)+v(i,j,k)*dz(k)
            end do
          end do
        end do

        do k=1,kbm1
          do j=2,jm
            do i=1,im
              v(i,j,k)=(v(i,j,k)-tps(i,j))+
     $                 (vtb(i,j)+vtf(i,j))/(dt(i,j)+dt(i,j-1))
            end do
          end do
        end do

! calculate w from u, v, dt (h+et), etf and etb
!      if (my_task .eq. 0) then       
!            write(6,112) "In advance-5:  ",  "uf(2,2,2)", uf(2,2,2),
!     $          "w(2,2,2)", w(2,2,2),          
!     $          "w(1,2,2)", w(1,2,2),          
!     $          "u(2,2,2)", u(2,2,2),          
!     $          "u(2,2,1)", u(2,2,1)
!      end if 
        call vertvl(dt,u,v,vfluxb,vfluxf,w,etf,etb)
!void vertvl_(float dt[][i_size], float u[][j_size][i_size],
!float v[][j_size][i_size],float vfluxb[][i_size],
!float vfluxf[][i_size], float w[][j_size][i_size],
!float etf[][i_size],float etb[][i_size]);

!      if (my_task .eq. 0) then       
!            write(6,112) "In advance-4:  ",  "uf(2,2,2)", uf(2,2,2),
!     $          "w(2,2,2)", w(2,2,2),          
!     $          "w(1,2,2)", w(1,2,2),          
!     $          "u(2,2,2)", u(2,2,2),          
!     $          "u(2,2,1)", u(2,2,1)
!      end if 
        call bcond(5)

        call exchange3d_mpi(w,im,jm,kb)

! set uf and vf to zero
        do k=1,kb
          do j=1,jm
            do i=1,im
              uf(i,j,k)=0.e0
              vf(i,j,k)=0.e0
            end do
          end do
        end do

! calculate q2f and q2lf using uf, vf, a and c as temporary variables
! 4.55% -->
!      if (my_task .eq. 0) then       
!            write(6,112) "In advance-3:  ",  "uf(2,2,2)", uf(2,2,2),
!     $          "w(2,2,2)", w(2,2,2),          
!     $          "w(1,2,2)", w(1,2,2),          
!     $          "u(2,2,2)", u(2,2,2),          
!     $          "u(2,2,1)", u(2,2,1)
!      end if 

!        call advq(q2b,q2,uf)
         call advq(q2b,q2,uf,u,dt,v,aam,w,etb,etf)

!advq(float qb[][j_size][i_size], float q[][j_size][i_size],
!float qf[][j_size][i_size], float u[][j_size][i_size],
!float dt[][i_size], float v[][j_size][i_size],
!float aam[][j_size][i_size], float w[][j_size][i_size],
!float etb[][i_size], float etf[][i_size]){

!        call advq(q2lb,q2l,vf)
        call advq(q2lb,q2l,vf,u,dt,v,aam,w,etb,etf)
!      if (my_task .eq. 0) then       
!            write(6,112) "In advance-2:  ",  "uf(2,2,2)", uf(2,2,2),
!     $          "w(2,2,2)", w(2,2,2),          
!     $          "w(1,2,2)", w(1,2,2),          
!     $          "u(2,2,2)", u(2,2,2),          
!     $          "u(2,2,1)", u(2,2,1)
!      end if 

! <-- 4.55%

! 11.87% -->
        call profq(etf,wusurf,wvsurf,wubot,wvbot,q2b,q2lb,u,v,
     $              km,uf,vf,q2,dt,kh,t,s,rho)
!void profq(float etf[][i_size], float wusurf[][i_size],
!float wvsurf[][i_size],float wubot[][i_size],
!float wvbot[][i_size], float q2b[][j_size][i_size],
!float q2lb[][j_size][i_size], float u[][j_size][i_size],
!float v[][j_size][i_size], float km[][j_size][i_size], 
!float uf[][j_size][i_size], float vf[][j_size][i_size],
!float q2[][j_size][i_size], float dt[][i_size],
!float kh[][j_size][i_size], float t[][j_size][i_size],
!float s[][j_size][i_size], float rho[][j_size][i_size]);
! <-- 11.87%

        call bcond(6)

        call exchange3d_mpi(uf(:,:,2:kbm1),im,jm,kbm2)
        call exchange3d_mpi(vf(:,:,2:kbm1),im,jm,kbm2)

        do k=1,kb
          do j=1,jm
            do i=1,im
              q2(i,j,k)=q2(i,j,k)
     $                   +.5e0*smoth*(uf(i,j,k)+q2b(i,j,k)
     $                                -2.e0*q2(i,j,k))
              q2l(i,j,k)=q2l(i,j,k)
     $                   +.5e0*smoth*(vf(i,j,k)+q2lb(i,j,k)
     $                                -2.e0*q2l(i,j,k))
              q2b(i,j,k)=q2(i,j,k)
              q2(i,j,k)=uf(i,j,k)
              q2lb(i,j,k)=q2l(i,j,k)
              q2l(i,j,k)=vf(i,j,k)
            end do
          end do
        end do
!      if (my_task .eq. 0) then       
!            write(6,112) "In advance-1:  ",  "uf(2,2,2)", uf(2,2,2),
!     $          "w(2,2,2)", w(2,2,2),          
!     $          "w(1,2,2)", w(1,2,2),          
!     $          "u(2,2,2)", u(2,2,2),          
!     $          "u(2,2,1)", u(2,2,1)
!      end if 

! calculate tf and sf using uf, vf, a and c as temporary variables
        if(mode.ne.4) then
          if(nadv.eq.1) then
            call advt1(tb,t,tclim,uf,dt,u,v,aam,w,etb,etf)
!void advt1(float fb[][j_size][i_size], float f[][j_size][i_size],
!float fclim[][j_size][i_size], float ff[][j_size][i_size],
!float dt[][i_size], float u[][j_size][i_size],
!float v[][j_size][i_size], float aam[][j_size][i_size],
!float w[][j_size][i_size], float etb[][i_size],
!float etf[][i_size]);
            call advt1(sb,s,sclim,vf,dt,u,v,aam,w,etb,etf)
          else if(nadv.eq.2) then
            call advt2(tb,t,tclim,uf,etb,u,v,etf,aam,w,dt)
            call advt2(sb,s,sclim,vf,etb,u,v,etf,aam,w,dt)
          else
            error_status=1
            write(6,'(/''Error: invalid value for nadv'')')
          end if


          call proft(uf,wtsurf,tsurf,nbct,etf,kh,swrad)
!void proft_(float f[][j_size][i_size], float wfsurf[][i_size],
!float fsurf[][i_size], int *f_nbc,
!float etf[][i_size], float kh[][j_size][i_size],
!float swrad[][i_size]);
          call proft(vf,wssurf,ssurf,nbcs,etf,kh,swrad)

          call bcond(4)

          call exchange3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
          call exchange3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)

          do k=1,kb
            do j=1,jm
              do i=1,im
                t(i,j,k)=t(i,j,k)
     $                    +.5e0*smoth*(uf(i,j,k)+tb(i,j,k)
     $                                 -2.e0*t(i,j,k))
                s(i,j,k)=s(i,j,k)
     $                    +.5e0*smoth*(vf(i,j,k)+sb(i,j,k)
     $                                 -2.e0*s(i,j,k))
                tb(i,j,k)=t(i,j,k)
                t(i,j,k)=uf(i,j,k)
                sb(i,j,k)=s(i,j,k)
                s(i,j,k)=vf(i,j,k)
              end do
            end do
          end do

          call dens(s,t,rho)
!void dens_(float si[][j_size][i_size], float ti[][j_size][i_size],
!float rhoo[][j_size][i_size], float fsm_f[][i_size],
!float *grav_f, float *rhoref_f, float *zz_f, float hh_f[][i_size]){

        end if

! calculate tracer
	if (tracer_flag.ne.0) then  
        do inb = 1,nb  !Loop thro the nb#tracers to be tracked
        tr3db(:,:,:) = trb(:,:,:,inb); tr3d(:,:,:) = tr(:,:,:,inb)
        vf(:,:,:)    = 0.0
	if (abs(tracer_flag).eq.1) then  !direct specification:
        call advt2_tr2(tr3db,tr3d,vf,a,c)
	endif
	if (abs(tracer_flag).eq.2) then  !source specification:
        call advt2_tr3(tr3db,tr3d,vf,a,c)
        endif
	rdisp2d(:,:)=0.0; k=1 !prescribe zero surface tracer flux
        call proft(vf,rdisp2d,rdisp2d,k,tps)
	call bcond(7) !tracer:
        call exchange3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)

! calculate uf and vf

        call advu(uf,w,u,advx,dt,v,egf,egb,e_atmos
     $ ,drhox,etb,ub,etf)

        call advv(vf,w,u,advy,dt,v,egf,egb,e_atmos
     $ ,drhoy,etb,vb,etf)
        call profu(etf,km,wusurf,uf,vb,ub,wubot)
        call profv(etf,km,wvsurf,vf,ub,vb,wvbot)

        call bcond(3)

        call exchange3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
        call exchange3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)

        do j=1,jm
          do i=1,im
            tps(i,j)=0.e0
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              tps(i,j)=tps(i,j)
     $                  +(uf(i,j,k)+ub(i,j,k)-2.e0*u(i,j,k))*dz(k)
            end do
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              u(i,j,k)=u(i,j,k)
     $                  +.5e0*smoth*(uf(i,j,k)+ub(i,j,k)
     $                               -2.e0*u(i,j,k)-tps(i,j))
            end do
          end do
        end do

        do j=1,jm
          do i=1,im
            tps(i,j)=0.e0
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              tps(i,j)=tps(i,j)
     $                  +(vf(i,j,k)+vb(i,j,k)-2.e0*v(i,j,k))*dz(k)
            end do
          end do
        end do

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              v(i,j,k)=v(i,j,k)
     $                  +.5e0*smoth*(vf(i,j,k)+vb(i,j,k)
     $                               -2.e0*v(i,j,k)-tps(i,j))
            end do
          end do
        end do

        do k=1,kb
          do j=1,jm
            do i=1,im
              ub(i,j,k)=u(i,j,k)
              u(i,j,k)=uf(i,j,k)
              vb(i,j,k)=v(i,j,k)
              v(i,j,k)=vf(i,j,k)
            end do
          end do
        end do

      end if
*/
	if ((iint != 1 || time0 != 0.0f) && mode != 2){
		/*
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = 0.0f;	
			}
		}
		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					tps[j][i] += u[k][j][i]*dz[k];	
				}
			}
		}
		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 1; i < im; i++){
					u[k][j][i] = (u[k][j][i]-tps[j][i])
								+(utb[j][i]+utf[j][i])
									/(dt[j][i]+dt[j][i-1]);
				}
			}
		}
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = 0.0f;	
			}
		}
		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					tps[j][i] += v[k][j][i]*dz[k];	
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 1; j < jm; j++){
				for (i = 0; i < im; i++){
					v[k][j][i] = (v[k][j][i]-tps[j][i])+(vtb[j][i]+vtf[j][i])/(dt[j][i]+dt[j-1][i]);
				}
			}
		}

#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

        //vertvl_(dt,u,v,vfluxb,vfluxf,w,etf,etb);//modified -w
        vertvl();

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		vertvl_time += time_consumed(&time_start_mode_internal,
									 &time_end_mode_internal);
#endif

//void vertvl_(float dt[][i_size], float u[][j_size][i_size],
//float v[][j_size][i_size],float vfluxb[][i_size],
//float vfluxf[][i_size], float w[][j_size][i_size],
//float etf[][i_size],float etb[][i_size]);
		
#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		bcond(5);//only modify w and reference fsm

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		bcond_time += time_consumed(&time_start_mode_internal,
									 &time_end_mode_internal);
#endif

		//exchange3d_mpi_xsz_(w,im,jm,0,kb-1);
		//exchange3d_mpi_bak(w,im,jm,0,kb-1);
		exchange3d_mpi(w,im,jm,kb);

		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = 0.0f;
					vf[k][j][i] = 0.0f;
				}
			}
		}

#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		//advq_(q2b,q2,uf,u,dt,v,aam,w,etb,etf);//only modify uf
		//
		//advq_(q2lb,q2l,vf,u,dt,v,aam,w,etb,etf);//only modify vf

		advq(q2b,q2,uf);
		advq(q2lb,q2l,vf);

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		advq_time += time_consumed(&time_start_mode_internal,
								   &time_end_mode_internal);
#endif


// modify:
//			uf, vf, 
//	+reference: kq, km, kh
//	+reference:	q2b, q2lb
#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		//profq_(etf,wusurf,wvsurf,wubot,wvbot,q2b,q2lb,u,v,
		//		km,uf,vf,q2,dt,kh,t,s,rho);//
		profq();

		bcond(6); //only modify uf vf

		//exchange3d_mpi_xsz_(uf,im,jm,1,kbm1-1);
		//exchange3d_mpi_bak(uf,im,jm,1,kbm1-1);
		exchange3d_mpi((float (*)[j_size][i_size])uf[1],im,jm,kbm1-1);
		//exchange3d_mpi_xsz_(vf,im,jm,1,kbm1-1);
		//exchange3d_mpi_bak(vf,im,jm,1,kbm1-1);
		exchange3d_mpi((float (*)[j_size][i_size])vf[1],im,jm,kbm1-1);

		
		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					q2[k][j][i] = q2[k][j][i]+0.5f*smoth*
												(uf[k][j][i]+q2b[k][j][i]-2.0f*q2[k][j][i]);	
					q2l[k][j][i] = q2l[k][j][i]+0.5f*smoth*
												(vf[k][j][i]+q2lb[k][j][i]-2.0f*q2l[k][j][i]);
					q2b[k][j][i] = q2[k][j][i];
					q2[k][j][i] = uf[k][j][i];
					q2lb[k][j][i] = q2l[k][j][i];
					q2l[k][j][i] = vf[k][j][i];
				}
			}
		}
		*/
		if (mode != 4){
			/*
			if (nadv == 1){
#ifndef TIME_DISABLE
				timer_now(&time_start_mode_internal);
#endif

				
				//advt1_(tb,t,tclim,uf,dt,u,v,aam,w,etb,etf);//modify -uf +t tb
				//advt1_(sb,s,sclim,vf,dt,u,v,aam,w,etb,etf);//modifuy -vf +s sb
				

				advt1(tb,t,tclim,uf,'T');
				advt1(sb,s,sclim,vf,'S');

#ifndef TIME_DISABLE
				timer_now(&time_end_mode_internal);
				advt1_time += time_consumed(&time_start_mode_internal,
							    			&time_end_mode_internal);
#endif
			}else if(nadv == 2){ 
#ifndef TIME_DISABLE
				timer_now(&time_start_mode_internal);
#endif

				//advt2_(tb,t,tclim,uf,etb,u,v,etf,aam,w,dt);//modify -uf +tb
				//advt2_(sb,s,sclim,vf,etb,u,v,etf,aam,w,dt);//modify -vf +sb

				advt2(tb,t,tclim,uf,'T');//modify -uf +tb +t
				advt2(sb,s,sclim,vf,'S');//modify -vf +sb +s

#ifndef TIME_DISABLE
				timer_now(&time_end_mode_internal);
				advt2_time += time_consumed(&time_start_mode_internal,
							    			&time_end_mode_internal);
#endif
			}else{
				error_status = 1;
				printf("Error: invalid value for nadv! Error find in File:%s, Func:%s, Line:%d\n",
						__FILE__, __func__, __LINE__);
			}





			//proft_(uf,wtsurf,tsurf,nbct,etf,kh,swrad);//modify +uf
			//proft_(vf,wssurf,ssurf,nbcs,etf,kh,swrad);//modify +vf

			proft(uf,wtsurf,tsurf,nbct);
			proft(vf,wssurf,ssurf,nbcs);


#ifndef TIME_DISABLE
			timer_now(&time_start_mode_internal);
#endif

			bcond(4);//modify -uf vf

#ifndef TIME_DISABLE
			timer_now(&time_end_mode_internal);
			bcond_time += time_consumed(&time_start_mode_internal,
									    &time_end_mode_internal);
#endif

			//exchange3d_mpi_xsz_(uf,im,jm,0,kbm1-1);
			//exchange3d_mpi_bak(uf,im,jm,0,kbm1-1);
			exchange3d_mpi(uf,im,jm,kbm1);
			
			//exchange3d_mpi_xsz_(vf,im,jm,0,kbm1-1);
			//exchange3d_mpi_bak(vf,im,jm,0,kbm1-1);
			exchange3d_mpi(vf,im,jm,kbm1);

			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						t[k][j][i] = t[k][j][i]+0.5f*smoth*
												(uf[k][j][i]+tb[k][j][i]-2.0f*t[k][j][i]);
						s[k][j][i] = s[k][j][i]+0.5f*smoth*
												(vf[k][j][i]+sb[k][j][i]-2.0f*s[k][j][i]);
						tb[k][j][i] = t[k][j][i];
						t[k][j][i] = uf[k][j][i];
						sb[k][j][i] = s[k][j][i];
						s[k][j][i] = vf[k][j][i];
					}
				}
			}

#ifndef TIME_DISABLE
			timer_now(&time_start_mode_internal);
#endif

			//dens_(s,t,rho);//modify -rho
			dens(s,t,rho);//modify -rho

#ifndef TIME_DISABLE
			timer_now(&time_end_mode_internal);
			dens_time += time_consumed(&time_start_mode_internal,
									   &time_end_mode_internal);
#endif
			*/
		}

		/*
		if (tracer_flag != 0){
			for (inb = 1; inb < nb; inb++){
				for (k = 0; k < kb; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < im; i++){
							tr3db[k][j][i] = trb[inb][k][j][i];	
							tr3d[k][j][i] = tr[inb][k][j][i];	
							vf[k][j][i] = 0;
						}
					}
				}
			

				if (ABS(tracer_flag) == 1){
					//advt2_tr2(tr3db, tr3d, vf);
				}else if (ABS(tracer_flag) == 2){
					//advtt2_tr3(tr3db, tr3d, vf);
				}

				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						rdisp2d[j][i] = 0;	
					}
				}

				//proft(vf, rdisp2d, rdisp2d, 1);
				//bcond(7);
				//exchange3d_mpi(vf, im, jm, kbm1);

				for (k = 0; k < kb; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < im; i++){
							tr[inb][k][j][i] = 
									tr[inb][k][j][i]
								   +0.5f*smoth*(vf[k][j][i]
											   +trb[inb][k][j][i]
											   -2.f*tr[inb][k][j][i]);
						}
					}
				}
			}
		}

#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		//modify -uf  //advx changed in advct
		//advu_(uf,w,u,advx,dt,v,egf,egb,e_atmos,drhox,etb,ub,etf);
		advu();

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		advu_time += time_consumed(&time_start_mode_internal,
								   &time_end_mode_internal);
#endif

#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		//modify -vf
		//advv_(vf,w,u,advy,dt,v,egf,egb,e_atmos,drhoy,etb,vb,etf);
		advv();

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		advv_time += time_consumed(&time_start_mode_internal,
								   &time_end_mode_internal);
#endif


#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		//profu_(etf,km,wusurf,uf,vb,ub,wubot);//modify +uf -wubot
		profu();

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		profu_time += time_consumed(&time_start_mode_internal,
								    &time_end_mode_internal);
#endif

#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		//profv_(etf,km,wvsurf,vf,ub,vb,wvbot);//modify +vf -wvbot
		profv();

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		profv_time += time_consumed(&time_start_mode_internal,
								    &time_end_mode_internal);
#endif

#ifndef TIME_DISABLE
		timer_now(&time_start_mode_internal);
#endif

		bcond(3);//modify -uf -vf	

#ifndef TIME_DISABLE
		timer_now(&time_end_mode_internal);
		bcond_time += time_consumed(&time_start_mode_internal,
									 &time_end_mode_internal);

#endif


		//exchange3d_mpi_xsz_(uf,im,jm,0,kbm1-1);
		//exchange3d_mpi_bak(uf,im,jm,0,kbm1-1);
		exchange3d_mpi(uf,im,jm,kbm1);
		
		//exchange3d_mpi_xsz_(vf,im,jm,0,kbm1-1);
		//exchange3d_mpi_bak(vf,im,jm,0,kbm1-1);
		exchange3d_mpi(vf,im,jm,kbm1);


		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = 0.0f;	
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					tps[j][i] = tps[j][i]+(uf[k][j][i]+ub[k][j][i]-2.0f*u[k][j][i])*dz[k];
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					u[k][j][i] = u[k][j][i]+0.5f*smoth*
											 (uf[k][j][i]+ub[k][j][i]-2.0f*u[k][j][i]-tps[j][i]);
				}
			}
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = 0.0f;	
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					tps[j][i] = tps[j][i]+(vf[k][j][i]+vb[k][j][i]-2.0f*v[k][j][i])*dz[k];
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					v[k][j][i] = v[k][j][i]+0.5f*smoth*
											 (vf[k][j][i]+vb[k][j][i]-2.0f*v[k][j][i]-tps[j][i]);
				}
			}
		}

		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					ub[k][j][i] = u[k][j][i];
					u[k][j][i] = uf[k][j][i];

					vb[k][j][i] = v[k][j][i];
					v[k][j][i] = vf[k][j][i];
				}
			}
		}
		*/
	}
	/*
      do j=1,jm
        do i=1,im
          egb(i,j)=egf(i,j)
          etb(i,j)=et(i,j)
          et(i,j)=etf(i,j)
          dt(i,j)=h(i,j)+et(i,j)
          utb(i,j)=utf(i,j)
          vtb(i,j)=vtf(i,j)
          vfluxb(i,j)=vfluxf(i,j)
        end do
      end do
	*/

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			egb[j][i] = egf[j][i];
			etb[j][i] = et[j][i];
			et[j][i] = etf[j][i];

			dt[j][i] = h[j][i]+et[j][i];

			utb[j][i] = utf[j][i];
			vtb[j][i] = vtf[j][i];

			vfluxb[j][i] = vfluxf[j][i];
		}
	}

	realvertvl();
	*/

	/*
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				f_u[k][j][i] = u[k][j][i];
				f_v[k][j][i] = v[k][j][i];
				f_w[k][j][i] = w[k][j][i];
				f_uf[k][j][i] = uf[k][j][i];
				f_vf[k][j][i] = vf[k][j][i];
				f_km[k][j][i] = km[k][j][i];
				f_kh[k][j][i] = kh[k][j][i];
				f_q2[k][j][i] = q2[k][j][i];
				f_q2b[k][j][i] = q2b[k][j][i];
				f_q2l[k][j][i] = q2l[k][j][i];
				f_q2lb[k][j][i] = q2lb[k][j][i];
				f_t[k][j][i] = t[k][j][i];
				f_tb[k][j][i] = tb[k][j][i];
				f_s[k][j][i] = s[k][j][i];
				f_sb[k][j][i] = sb[k][j][i];
				f_rho[k][j][i] = rho[k][j][i];

				f_ub[k][j][i] = ub[k][j][i];
				f_vb[k][j][i] = vb[k][j][i];
			}
		}
	}
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_egb[j][i] = egb[j][i];
			f_etb[j][i] = etb[j][i];
			f_et[j][i] = et[j][i];
			f_dt[j][i] = dt[j][i];
			f_utb[j][i] = utb[j][i];
			f_vtb[j][i] = vtb[j][i];
			f_vfluxb[j][i] = vfluxb[j][i];
			f_vfluxf[j][i] = vfluxf[j][i];
			f_wubot[j][i] = wubot[j][i];
			f_wvbot[j][i] = wvbot[j][i];
		}
	}
	*/
	
	return;
}

/*
      subroutine store_mean

      implicit none
      include 'pom.h'

      if (mode.eq.2) then !lyo:channel:
      uab_mean    = uab_mean    + cor     !uab
      vab_mean    = vab_mean    + aam2d   !vab
      elb_mean    = elb_mean    + e_atmos !elb
      else
      uab_mean    = uab_mean    + uab
      vab_mean    = vab_mean    + vab
      elb_mean    = elb_mean    + elb
      endif
      wusurf_mean = wusurf_mean + wusurf
      wvsurf_mean = wvsurf_mean + wvsurf
      wtsurf_mean = wtsurf_mean + wtsurf
      wssurf_mean = wssurf_mean + wssurf
      u(:,:,kb)   = wubot(:,:)        !fhx:20110318:store wvbot
      v(:,:,kb)   = wvbot(:,:)        !fhx:20110318:store wvbot
      u_mean      = u_mean      + u + ustks !exp347:add ustks
      v_mean      = v_mean      + v + vstks !exp347:add vstks
      w(:,:,kb)   = ustks(:,:,1)      !lyo:!stokes:
      w_mean      = w_mean      + w
      t_mean      = t_mean      + t
      s_mean      = s_mean      + s
      rho_mean    = rho_mean    + rho
!     kh(:,:,kb)  = vstks(:,:,1)      !lyo:!stokes:
      kh_mean     = kh_mean     + kh
      aam(:,:,kb) = cbc(:,:)          !lyo:20110315:botwavedrag:store cbc
      km_mean     = km_mean     + aam !lyo:20110202:save aam inst. of km
!     tr_mean     = tr_mean     + tr  !fhx:tracer

      num = num + 1

      return
      end
*/

void store_mean(){
	
	int i, j, k;
	if (mode == 2){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uab_mean[j][i] += cor[j][i];	
				vab_mean[j][i] += aam2d[j][i];	
				elb_mean[j][i] += e_atmos[j][i];	
			}
		}
	}else{
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uab_mean[j][i] += uab[j][i];
				vab_mean[j][i] += vab[j][i];
				elb_mean[j][i] += elb[j][i];
			}
		}
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			wusurf_mean[j][i] += wusurf[j][i];
			wvsurf_mean[j][i] += wvsurf[j][i];
			wtsurf_mean[j][i] += wtsurf[j][i];
			wssurf_mean[j][i] += wssurf[j][i];
			u[kb-1][j][i] = wubot[j][i];
			v[kb-1][j][i] = wvbot[j][i];
			w[kb-1][j][i] = ustks[0][j][i];
			aam[kb-1][j][i] = cbc[j][i];
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				u_mean[k][j][i] = u_mean[k][j][i]+u[k][j][i]+ustks[k][j][i];
				v_mean[k][j][i] = v_mean[k][j][i]+v[k][j][i]+vstks[k][j][i];
				w_mean[k][j][i] += w[k][j][i];
				t_mean[k][j][i] += t[k][j][i];
				s_mean[k][j][i] += s[k][j][i];
				rho_mean[k][j][i] += rho[k][j][i];
				kh_mean[k][j][i] += kh[k][j][i];
				km_mean[k][j][i] += aam[k][j][i];

			}
		}
	}

	num += 1;
	return;
}

/*
      subroutine store_surf_mean !fhx:20110131:add new subr. for surf mean

      implicit none
      include 'pom.h'

      if (mode.eq.2) then !lyo:channel:
      usrf_mean    = usrf_mean    + uab
      vsrf_mean    = vsrf_mean    + vab
      else
      usrf_mean    = usrf_mean    + u(:,:,1)
      vsrf_mean    = vsrf_mean    + v(:,:,1)
      endif
      if (n_east.eq.-1 .and. n_north.eq.-1) then
         elb(im,jm)=cor(im,jm) !store f-value useful if f-plane
         endif
      elsrf_mean    = elsrf_mean    + elb

      if ( calc_wind ) then !lyo:channel:
      uwsrf_mean = uwsrf_mean + uwsrf
      vwsrf_mean = vwsrf_mean + vwsrf      
      else
      uwsrf_mean = uwsrf_mean + wusurf !store -u_windstress for idealized case
      vwsrf_mean = vwsrf_mean + wvsurf !store -v_windstress
      endif

!     utf_mean    = utf_mean    + e_atmos
      utf_mean    = utf_mean    + utf !lyo:!vort:U-transport!part of CPVF:
      vtf_mean    = vtf_mean    + vtf !Should be (ua,va)D@isplit; but approx.ok

      xstks_mean = xstks_mean + xstks !lyo:!stokes:
      ystks_mean = ystks_mean + ystks !lyo:!stokes:

!lyo:!vort:beg:Write vorticity(e.g. JEBAR) analysis
      if ( calc_vort ) then
      celg_mean = celg_mean+celg;  ctsurf_mean=ctsurf_mean+ctsurf;
      ctbot_mean=ctbot_mean+ctbot; cpvf_mean  =  cpvf_mean+cpvf;
      cjbar_mean=cjbar_mean+cjbar; cadv_mean  =  cadv_mean+cadv;
      cten_mean = cten_mean+cten;
      endif
!lyo:!vort:end:Write vorticity(e.g. JEBAR) analysis

      nums = nums + 1

      return
      end
*/

void store_surf_mean(){
	int i, j, k;

	if (mode == 2){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				usrf_mean[j][i] += uab[j][i];	
				vsrf_mean[j][i] += vab[j][i];	
			}
		}
	}else{
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				usrf_mean[j][i] += u[0][j][i];	
				vsrf_mean[j][i] += v[0][j][i];	
			}
		}
	}

	if (n_east == -1 && n_north == -1){
		elb[jm-1][im-1] = cor[jm-1][im-1];	
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			elsrf_mean[j][i] += elb[j][i];	
		}
	}

	if (calc_wind){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uwsrf_mean[j][i] += uwsrf[j][i];	
				vwsrf_mean[j][i] += vwsrf[j][i];	
			}
		}
	}else{
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uwsrf_mean[j][i] += wusurf[j][i];	
				vwsrf_mean[j][i] += wvsurf[j][i];	
			}
		}
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			utf_mean[j][i] += utf[j][i];	
			vtf_mean[j][i] += vtf[j][i];	
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				xstks_mean[k][j][i] += xstks[k][j][i];	
				ystks_mean[k][j][i] += ystks[k][j][i];	
			}
		}
	}

	if (calc_vort){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				celg_mean[j][i] += celg[j][i];	
				ctsurf_mean[j][i] += ctsurf[j][i];	
				ctbot_mean[j][i] += ctbot[j][i];	
				cpvf_mean[j][i] += cpvf[j][i];	
				cjbar_mean[j][i] += cjbar[j][i];	
				cadv_mean[j][i] += cadv[j][i];	
				cten_mean[j][i] += cten[j][i];	
			}
		}
	}
	nums += 1;

	return;
}

/*
void print_section_(float *f_area_tot, float *f_vol_tot,
					float *f_d_area, float *f_d_vol,
					float *f_elev_ave, float *f_temp_ave,
					float *f_salt_ave){
*/

void print_section(){
		
      int i,j,k;
      float area_tot,vol_tot,d_area,d_vol;
      float elev_ave, temp_ave, salt_ave;

/*
      if(mod(iint,iprint).eq.0) then

! print time
        if(my_task.eq.master_task) write(6,'(/
     $    ''**********************************************************''
     $    /''time ='',f9.4,'', iint ='',i8,'', iext ='',i8,
     $    '', iprint ='',i8)') time,iint,iext,iprint

! check for errors
        call sum0d_mpi(error_status,master_task)
        call bcast0d_mpi(error_status,master_task)
        if(error_status.ne.0) then
          if(my_task.eq.master_task) write(*,'(/a)')
     $                                       'POM terminated with error
     $   in advance.f'
          call finalize_mpi
          stop
        end if

! local averages
        vol_tot  = 0.e0
        area_tot = 0.e0
        temp_ave = 0.e0
        salt_ave = 0.e0
        elev_ave = 0.e0

        do k=1,kbm1
           do j=1,jm
              do i=1,im
                 d_area   = dx(i,j) * dy(i,j)
                 d_vol    = d_area * dt(i,j) * dz(k) * fsm(i,j)
                 vol_tot  = vol_tot + d_vol
                 temp_ave = temp_ave + tb(i,j,k)*d_vol
                 salt_ave = salt_ave + sb(i,j,k)*d_vol
              end do
           end do
        end do

        do j=1,jm
          do i=1,im
             d_area = dx(i,j) * dy(i,j)
             area_tot = area_tot + d_area
             elev_ave = elev_ave + et(i,j) * d_area
          end do
        end do


        call sum0d_mpi( temp_ave, master_task )
        call sum0d_mpi( salt_ave, master_task )
        call sum0d_mpi( elev_ave, master_task )
        call sum0d_mpi(  vol_tot, master_task )
        call sum0d_mpi( area_tot, master_task )

        temp_ave = temp_ave / vol_tot
        salt_ave = salt_ave / vol_tot
        elev_ave = elev_ave / area_tot

! print averages
        if(my_task.eq.master_task) 
     $       write(*,'(a,e15.8,2(a,f11.8),a)') 
     $       "mean ; et = ",elev_ave," m, tb = ",
     $       temp_ave + tbias," deg, sb = ",
     $       salt_ave + sbias ," psu"
		end if

      return
      end

*/
	if (iint%iprint == 0){
		if (my_task == master_task){
			printf("*******************************************************\n");
			printf("time = %f, iint = %d, iext = %d, iprint = %d\n", 
					model_time, iint, iext, iprint);
		}
		sum0i_mpi(&error_status, master_task);
		bcast0d_mpi(&error_status, master_task);

		if (error_status != 0){
			printf("POM terminated with error! in file: %s, function: %s, line: %d\n",
					__FILE__, __func__, __LINE__);
			finalize_mpi();
		}
		
		vol_tot = 0.0f;
		area_tot = 0.0f;
		temp_ave = 0.0f;
		salt_ave = 0.0f;
		elev_ave = 0.0f;

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					d_area = dx[j][i]*dy[j][i];
					d_vol = d_area*dt[j][i]*dz[k]*fsm[j][i];
					vol_tot = vol_tot + d_vol;
					temp_ave = temp_ave + tb[k][j][i]*d_vol;
					salt_ave = salt_ave + sb[k][j][i]*d_vol;
				}
			}
		} 

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				d_area = dx[j][i]*dy[j][i];
				area_tot += d_area;
				elev_ave += et[j][i]*d_area;
			}
		}

		sum0f_mpi(&temp_ave, master_task);
		sum0f_mpi(&salt_ave, master_task);
		sum0f_mpi(&elev_ave, master_task);
		sum0f_mpi(&vol_tot, master_task);
		sum0f_mpi(&area_tot, master_task);


		temp_ave /= vol_tot;
		salt_ave /= vol_tot;
		elev_ave /= area_tot;

		if (my_task == master_task){
			printf("****************************************************\n");
			printf("et = %+30.24e m, tb = %+30.24e deg, sb = %+30.24e psu\n",
					elev_ave, temp_ave+tbias, salt_ave+sbias);	

			printf("****************************************************\n");
#ifndef TIME_DISABLE
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_advance",
					advance_time/1.E+6, 100.f*advance_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_momentum3d",
					momentum3d_time/1.E+6, 100.f*momentum3d_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_mode_interaction",
					mode_interaction_time/1.E+6, 100.f*mode_interaction_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_mode_external",
					mode_external_time/1.E+6, 100.f*mode_external_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_mode_internal",
					mode_internal_time/1.E+6, 100.f*mode_internal_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_store_mean",
					store_mean_time/1.E+6, 100.f*store_mean_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_store_surf_mean",
					store_surf_mean_time/1.E+6, 100.f*store_surf_mean_time/advance_time);

			printf("*******************************************************\n");

			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advave",
					advave_time/1.E+6, 100.f*advave_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advct",
					advct_time/1.E+6, 100.f*advct_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advq",
					advq_time/1.E+6, 100.f*advq_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advt1",
					advt1_time/1.E+6, 100.f*advt1_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advt2",
					advt2_time/1.E+6, 100.f*advt2_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advu",
					advu_time/1.E+6, 100.f*advu_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_advv",
					advv_time/1.E+6, 100.f*advv_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_baropg",
					baropg_time/1.E+6, 100.f*baropg_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_dens",
					dens_time/1.E+6, 100.f*dens_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_profq",
					profq_time/1.E+6, 100.f*profq_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_proft",
					proft_time/1.E+6, 100.f*proft_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_profu",
					profu_time/1.E+6, 100.f*profu_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_profv",
					profv_time/1.E+6, 100.f*profv_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_smol_adif",
					smol_adif_time/1.E+6, 100.f*smol_adif_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_vertvl",
					vertvl_time/1.E+6, 100.f*vertvl_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_solver_realvertvl",
					realvertvl_time/1.E+6, 100.f*realvertvl_time/advance_time);

			printf("*******************************************************\n");

			printf("%-30s:  %7.3lf; %7.3f%\n", "time_3d_mpi",
					exchange3d_mpi_time/1.E+6, 100.f*exchange3d_mpi_time/advance_time);
			printf("%-30s:  %7.3lf; %7.3f%\n", "time_2d_mpi",
					exchange2d_mpi_time/1.E+6, 100.f*exchange2d_mpi_time/advance_time);
			printf("*******************************************************\n\n\n");
#endif
		}
	}
/*
      call call_time_end(print_section_time_end_xsz)
      print_section_time_xsz = print_section_time_end_xsz -
     $                         print_section_time_start_xsz +
     $                         print_section_time_xsz

      time_total_xsz = print_section_time_end_xsz -
     $                 time_start_xsz
*/

	return;
}


void check_velocity(){
//! check if velocity condition is violated

    float vamax,atot,darea,dvol,eaver,saver,taver,vtot,tsalt;
	int i,j,k;
	int imax, jmax;

	vamax=0.0f;

/*
      do j=1,jm
        do i=1,im
          if(abs(vaf(i,j)).ge.vamax) then
            vamax=abs(vaf(i,j))
            imax=i
            jmax=j
          end if
        end do
      end do
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			if (ABS(vaf[j][i]) >= vamax){
				vamax = ABS(vaf[j][i]);	
				imax = i;
				jmax = j;
			}
		}
	}

/*
      if(vamax.gt.vmaxl) then
        if(my_task.eq.master_task.and.error_status.eq.0) write(6,'(/
     $    ''Error: velocity condition violated''/''time ='',f9.4,
     $    '', iint ='',i8,'', iext ='',i8,'', iprint ='',i8,/
     $    ''vamax ='',e12.3,''   imax,jmax ='',2i5)')
     $    time,iint,iext,iprint,vamax,imax,jmax
        error_status=1
      end if
*/
	if (vamax > vmaxl){
		if (my_task == master_task && error_status == 0){
			printf("Error: velocity condition violated! time = %f, iint = %d, \
					iext = %d, iprint = %d, vamax = %f, imax = %d, jmax = %d\n",
					time, iint, iext, iprint, vamax, imax, jmax);	
			error_status = 1;
		}
	}

/*
      call call_time_end(check_velocity_time_end_xsz)
      check_velocity_time_xsz = check_velocity_time_end_xsz - 
     $                          check_velocity_time_start_xsz +
     $                          check_velocity_time_xsz

*/
    return;
}
