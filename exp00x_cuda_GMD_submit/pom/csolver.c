#include<mpi.h>
#include<stdio.h>
#include<math.h>
#include<sys/time.h>

#include"csolver.h"
#include"cparallel_mpi.h"

void dens(float si[][j_size][i_size], 
		  float ti[][j_size][i_size],
		  float rhoo[][j_size][i_size]){

//! calculate (density-1000.)/rhoref.
//! see: Mellor, G.L., 1991, J. Atmos. Oceanic Tech., 609-611
//! note: if pressure is not used in dens, buoyancy term (boygr) in
//! subroutine profq must be changed (see note in subroutine profq)

	int i,j,k;
    float cr,p,rhor,sr,tr,tr2,tr3,tr4;
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				
				tr = ti[k][j][i] + tbias;
				sr = si[k][j][i] + sbias;
				tr2 = tr*tr;
				tr3 = tr2*tr;
				tr4 = tr3*tr;
//! approximate pressure in units of bars
				p = grav*rhoref*(-zz[k]*h[j][i])*1.0e-5f;

				
				rhor=-0.157406e0f+6.793952e-2f*tr
			           -9.095290e-3f*tr2+1.001685e-4f*tr3
			           -1.120083e-6f*tr4+6.536332e-9f*tr4*tr;

				
				rhor=rhor+(0.824493e0f-4.0899e-3f*tr
		                +7.6438e-5f*tr2-8.2467e-7f*tr3
						+5.3875e-9f*tr4)*sr
						+(-5.72466e-3f+1.0227e-4f*tr
						-1.6546e-6f*tr2)*powf(ABS(sr), 1.5f)
						+4.8314e-4f*sr*sr;

				cr=1449.1e0f+0.0821e0f*p+4.55e0f*tr-0.045e0f*tr2
                      +1.34e0f*(sr-35.0e0f);

				rhor=rhor+1.0e5f*p/(cr*cr)*(1.0e0f-2.e0f*p/(cr*cr));

				rhoo[k][j][i]=rhor/rhoref*fsm[j][i];
				
			}
		}
	}
      
	return;
}

//!fhx:Toni:npg
/*
void baropg_mcc(float rho[][j_size][i_size], 
				float rmean[][j_size][i_size], 
				float d[][i_size],
				float dum[][i_size], 
				float dvm[][i_size],
				float dt[][i_size],
				float drhox[][j_size][i_size],
				float drhoy[][j_size][i_size],
				float ramp){
*/
void baropg_mcc(){

/*change:drhox,drhoy, rho(in fact no).
 *
 *
 *
 */
/*
 *rho: used in other functions in solver.f
 *
 *drhox: write in baropg of solver.f and referenced in advu
 *
 *dt: write again in advance.f dt = h + et
 *
 *drhoy: write in baropg of solver.f and referenced in advv
 *
 *ramp: it is always 1 in this program, but we shoud reference it
 */

//! calculate  baroclinic pressure gradient
//! 4th order correction terms, following McCalpin
	int i, j, k;	
	float d4[j_size][i_size], ddx[j_size][i_size],
		  drho[k_size][j_size][i_size], rhou[k_size][j_size][i_size];
	float rho4th[k_size][j_size+1][i_size+1], 
		  d4th[j_size+1][i_size+1];

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rho[k][j][i] -= rmean[k][j][i];	
			}
		}
	}

//! convert a 2nd order matrices to special 4th order
//! special 4th order case
    order2d_mpi(d,d4th,im,jm);
    order3d_mpi(rho,rho4th,im,jm,kb);

//! compute terms correct to 4th order
	for (i = 0; i < im; i++){
		for (j = 0; j < jm; j++){
			ddx[j][i] = 0;
			d4[j][i] = 0;
		}
	}
	
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rhou[k][j][i] = 0;	
				drho[k][j][i] = 0;
			}
		}
	}

//! compute DRHO, RHOU, DDX and D4
	for (j = 0; j < jm; j++){
		for (i = 1; i < im; i++){
			for (k = 1; k < kbm1; k++){
				drho[k][j][i] = (rho[k][j][i]-rho[k][j][i-1])*dum[j][i];
				rhou[k][j][i] = 0.5f*(rho[k][j][i]+rho[k][j][i-1])*dum[j][i];
			}
			ddx[j][i] = (d[j][i]-d[j][i-1])*dum[j][i];
			d4[j][i] = 0.5f*(d[j][i]+d[j][i-1])*dum[j][i];
		}
	}

	if (n_west == -1){
		for (j = 0; j < jm; j++){
			for (i = 2; i < imm1; i++){
				for (k = 0; k < kbm1; k++){
					drho[k][j][i] = drho[k][j][i] 
								   -(1.0f/24.0f)
									*(dum[j][i+1]*(rho[k][j][i+1]
												  -rho[k][j][i]) 
								     -2.0f*(rho[k][j][i]
									       -rho[k][j][i-1])
									 +dum[j][i-1]*(rho[k][j][i-1]
												  -rho[k][j][i-2]));
					rhou[k][j][i] = rhou[k][j][i] 
								   +(1.0f/16.0f)*
									 (dum[j][i+1]*(rho[k][j][i]
												  -rho[k][j][i+1])
									 +dum[j][i-1]*(rho[k][j][i-1]
												  -rho[k][j][i-2]));
				}
				ddx[j][i] = ddx[j][i]
						   -(1.0f/24.0f)
							*(dum[j][i+1]*(d[j][i+1]-d[j][i])
							 -2.0f*(d[j][i]-d[j][i-1])
							 +dum[j][i-1]*(d[j][i-1]-d[j][i-2]));
				d4[j][i] = d4[j][i]
						  +(1.0f/16.0f)
						   *(dum[j][i+1]*(d[j][i]-d[j][i+1])
						    +dum[j][i-1]*(d[j][i-1]-d[j][i-2]));
			}
		}
	}else{
		for (j = 0; j < jm; j++){
			for (i = 1; i < imm1; i++){
				for (k = 0; k < kbm1; k++){
					drho[k][j][i] = drho[k][j][i] 
								   -(1.0f/24.0f)
									*(dum[j][i+1]*(rho[k][j][i+1]
												  -rho[k][j][i]) 
								     -2.0f*(rho[k][j][i]
									       -rho[k][j][i-1])
									 +dum[j][i-1]*(rho[k][j][i-1]
												  -rho4th[k][j][i-1]));
					rhou[k][j][i] = rhou[k][j][i] 
								   +(1.0f/16.0f)*
									 (dum[j][i+1]*(rho[k][j][i]
												  -rho[k][j][i+1])
									 +dum[j][i-1]*(rho[k][j][i-1]
												  -rho4th[k][j][i-1]));
				}
				ddx[j][i] = ddx[j][i]
						   -(1.0f/24.0f)
							*(dum[j][i+1]*(d[j][i+1]-d[j][i])
							 -2.0f*(d[j][i]-d[j][i-1])
							 +dum[j][i-1]*(d[j][i-1]-d4th[j][i-2]));
				d4[j][i] = d4[j][i]
						  +(1.0f/16.0f)
						   *(dum[j][i+1]*(d[j][i]-d[j][i+1])
						    +dum[j][i-1]*(d[j][i-1]-d4th[j][i-2]));
			}
		}
	}

//! calculate x-component of baroclinic pressure gradient
	
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			drhox[0][j][i] = grav*(-zz[0])*d4[j][i]*drho[0][j][i];
		}
	}

	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i] = drhox[k-1][j][i]
								+grav*0.5f*dzz[k-1]*d4[j][i]
								 *(drho[k-1][j][i]+drho[k][j][i])
								+grav*0.5f*(zz[k-1]+zz[k])*ddx[j][i]
								 *(rhou[k][j][i]-rhou[k-1][j][i]);
			}
		}
	}

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i]	 = 0.25f*(dt[j][i]+dt[j][i-1])
				                        *drhox[k][j][i]*dum[j][i]
										*(dy[j][i]+dy[j][i-1]);
			}
		}
	}

//! compute terms correct to 4th order
	
	for (i = 0; i < im; i++){
		for (j = 0; j < jm; j++){
			ddx[j][i] = 0;
			d4[j][i] = 0;
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rhou[k][j][i] = 0;
				drho[k][j][i] = 0;
			}
		}
	}

//! compute DRHO, RHOU, DDX and D4
	for (j = 1; j < jm; j++){
		for (i = 0; i < im; i++){
			for (k = 0; k < kbm1; k++){
				drho[k][j][i] = (rho[k][j][i]-rho[k][j-1][i])*dvm[j][i];
				rhou[k][j][i] = 0.5f*(rho[k][j][i]+rho[k][j-1][i])*dvm[j][i];
			}
			ddx[j][i] = (d[j][i]-d[j-1][i])*dvm[j][i];
			d4[j][i] = 0.5f*(d[j][i]+d[j-1][i])*dvm[j][i];
		}
	}

	if (n_south == -1){
		for (j = 2; j < jmm1; j++){
			for (i = 0; i < im; i++){
				for (k = 0; k < kbm1; k++){
					drho[k][j][i] = drho[k][j][i] 
								   -(1.0f/24.0f)
									*(dvm[j+1][i]*(rho[k][j+1][i]
												  -rho[k][j][i])
									 -2.0f*(rho[k][j][i]-rho[k][j-1][i])
									 +dvm[j-1][i]*(rho[k][j-1][i]
												  -rho[k][j-2][i]));
					rhou[k][j][i] = rhou[k][j][i]
								   +(1.0f/16.0f)
									*(dvm[j+1][i]*(rho[k][j][i]
												  -rho[k][j+1][i])
									 +dvm[j-1][i]*(rho[k][j-1][i]
												  -rho[k][j-2][i]));
				}
				ddx[j][i] = ddx[j][i]
						   -(1.0f/24)
						    *(dvm[j+1][i]*(d[j+1][i]-d[j][i])
							 -2.0f*(d[j][i]-d[j-1][i])
							 +dvm[j-1][i]*(d[j-1][i]-d[j-2][i]));
				d4[j][i] = d4[j][i]
						  +(1.0f/16)
						   *(dvm[j+1][i]*(d[j][i]-d[j+1][i])
						    +dvm[j-1][i]*(d[j-1][i]-d[j-2][i]));
			}
		}
	}else{
		for (j = 1; j < jmm1; j++){
			for (i = 0; i < im; i++){
				for (k = 0; k < kbm1; k++){
					drho[k][j][i] = drho[k][j][i] 
								   -(1.0f/24.0f)
									*(dvm[j+1][i]*(rho[k][j+1][i]
												  -rho[k][j][i])
									 -2.0f*(rho[k][j][i]-rho[k][j-1][i])
									 +dvm[j-1][i]*(rho[k][j-1][i]
												  -rho4th[k][j-1][i]));
					rhou[k][j][i] = rhou[k][j][i]
								   +(1.0f/16.0f)
									*(dvm[j+1][i]*(rho[k][j][i]
												  -rho[k][j+1][i])
									 +dvm[j-1][i]*(rho[k][j-1][i]
												  -rho4th[k][j-1][i]));
				}
				ddx[j][i] = ddx[j][i]
						   -(1.0f/24)
						    *(dvm[j+1][i]*(d[j+1][i]-d[j][i])
							 -2.0f*(d[j][i]-d[j-1][i])
							 +dvm[j-1][i]*(d[j-1][i]-d4th[j-2][i]));
				d4[j][i] = d4[j][i]
						  +(1.0f/16)
						   *(dvm[j+1][i]*(d[j][i]-d[j+1][i])
						    +dvm[j-1][i]*(d[j-1][i]-d4th[j-2][i]));
			}
		}

	}
//! calculate y-component of baroclinic pressure gradient

	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			drhoy[0][j][i] = grav*(-zz[0])*d4[j][i]*drho[0][j][i];	
		}
	}

	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhoy[k][j][i] = drhoy[k-1][j][i] 
								+grav*0.5f*dzz[k-1]*d4[j][i]
								 *(drho[k-1][j][i]+drho[k][j][i])
								+grav*0.5f*(zz[k-1]+zz[k])*ddx[j][i]
								 *(rhou[k][j][i]-rhou[k-1][j][i]);
			}
		}
	}


	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhoy[k][j][i] = 0.25f*(dt[j][i]+dt[j-1][i])
									  *drhoy[k][j][i]*dvm[j][i]
									  *(dx[j][i]+dx[j-1][i]);
			}
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i] *= ramp;
				drhoy[k][j][i] *= ramp;
			}
		}
	}

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rho[k][j][i] = rho[k][j][i] + rmean[k][j][i];	
			}
		}
	}

	return;
}


/*
void baropg(float rho[][j_size][i_size], 
			float rmean[][j_size][i_size], 
		    float dum[][i_size], 
		    float dvm[][i_size],
			float dt[][i_size], 
			float drhox[][j_size][i_size],
			float drhoy[][j_size][i_size], 
			float ramp){
*/

void baropg(){

/*change:drhox,drhoy, rho(in fact no).
 *
 *
 *
 */
/*
 *rho: used in other functions in solver.f
 *
 *drhox: write in baropg_mcc of solver.f and referenced in advu
 *
 *dt: write again in advance.f dt = h + et
 *
 *drhoy: write in baropg_mcc of solver.f and referenced in advv
 *
 *ramp: it is always 1 in this program, but we shoud reference it
 */
	int i,j,k;
/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            rho(i,j,k)=rho(i,j,k)-rmean(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rho[k][j][i]-=rmean[k][j][i];	
			}
		}
	}

//! calculate x-component of baroclinic pressure gradient
/*
      do j=2,jmm1
        do i=2,imm1
          drhox(i,j,1)=.5e0*grav*(-zz(1))*(dt(i,j)+dt(i-1,j))
     $                  *(rho(i,j,1)-rho(i-1,j,1))
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			drhox[0][j][i]=0.5f*grav*(-zz[0])*(dt[j][i]+dt[j][i-1])
							   *(rho[0][j][i]-rho[0][j][i-1]);
		}
	}
/*
      do k=2,kbm1
        do j=2,jmm1
          do i=2,imm1
            drhox(i,j,k)=drhox(i,j,k-1)
     $                    +grav*.25e0*(zz(k-1)-zz(k))
     $                      *(dt(i,j)+dt(i-1,j))
     $                      *(rho(i,j,k)-rho(i-1,j,k)
     $                        +rho(i,j,k-1)-rho(i-1,j,k-1))
     $                    +grav*.25e0*(zz(k-1)+zz(k))
     $                      *(dt(i,j)-dt(i-1,j))
     $                      *(rho(i,j,k)+rho(i-1,j,k)
     $                        -rho(i,j,k-1)-rho(i-1,j,k-1))
          end do
        end do
      end do
*/
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i]=drhox[k-1][j][i]
								+grav*0.25f*(zz[k-1]-zz[k])
									*(dt[j][i]+dt[j][i-1])
									*(rho[k][j][i]-rho[k][j][i-1]
										+rho[k-1][j][i]-rho[k-1][j][i-1])
								+grav*0.25f*(zz[k-1]+zz[k])
									*(dt[j][i]-dt[j][i-1])
									*(rho[k][j][i]+rho[k][j][i-1]
										-rho[k-1][j][i]-rho[k-1][j][i-1]);
			}
		}
	}

/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            drhox(i,j,k)=.25e0*(dt(i,j)+dt(i-1,j))
     $                        *drhox(i,j,k)*dum(i,j)
     $                        *(dy(i,j)+dy(i-1,j))
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i]=0.25f*(dt[j][i]+dt[j][i-1])
								   *drhox[k][j][i]*dum[j][i]
								   *(dy[j][i]+dy[j][i-1]);
			}
		}
	}
//! calculate y-component of baroclinic pressure gradient
/*
      do j=2,jmm1
        do i=2,imm1
          drhoy(i,j,1)=.5e0*grav*(-zz(1))*(dt(i,j)+dt(i,j-1))
     $                  *(rho(i,j,1)-rho(i,j-1,1))
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			drhoy[0][j][i]=0.5f*grav*(-zz[0])*(dt[j][i]+dt[j-1][i])
							   *(rho[0][j][i]-rho[0][j-1][i]);
		}
	}

/*
      do k=2,kbm1
        do j=2,jmm1
          do i=2,imm1
            drhoy(i,j,k)=drhoy(i,j,k-1)
     $                    +grav*.25e0*(zz(k-1)-zz(k))
     $                      *(dt(i,j)+dt(i,j-1))
     $                      *(rho(i,j,k)-rho(i,j-1,k)
     $                        +rho(i,j,k-1)-rho(i,j-1,k-1))
     $                    +grav*.25e0*(zz(k-1)+zz(k))
     $                      *(dt(i,j)-dt(i,j-1))
     $                      *(rho(i,j,k)+rho(i,j-1,k)
     $                        -rho(i,j,k-1)-rho(i,j-1,k-1))
          end do
        end do
      end do
*/
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1;i < imm1; i++){
				drhoy[k][j][i]=drhoy[k-1][j][i]	
								+grav*0.25f*(zz[k-1]-zz[k])
									 *(dt[j][i]+dt[j-1][i])
									 *(rho[k][j][i]-rho[k][j-1][i]
										+rho[k-1][j][i]-rho[k-1][j-1][i])
								+grav*0.25f*(zz[k-1]+zz[k])
									 *(dt[j][i]-dt[j-1][i])
									 *(rho[k][j][i]+rho[k][j-1][i]
										-rho[k-1][j][i]-rho[k-1][j-1][i]);
			}
		}
	}

/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            drhoy(i,j,k)=.25e0*(dt(i,j)+dt(i,j-1))
     $                        *drhoy(i,j,k)*dvm(i,j)
     $                        *(dx(i,j)+dx(i,j-1))
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhoy[k][j][i]=0.25f*(dt[j][i]+dt[j-1][i])
									*drhoy[k][j][i]*dvm[j][i]
									*(dx[j][i]+dx[j-1][i]);
			}
		}
	}

/*
      do k=1,kb
        do j=2,jmm1
          do i=2,imm1
            drhox(i,j,k)=ramp*drhox(i,j,k)
            drhoy(i,j,k)=ramp*drhoy(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				drhox[k][j][i] *= (ramp);	
				drhoy[k][j][i] *= (ramp);	
			}
		}
	}

/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            rho(i,j,k)=rho(i,j,k)+rmean(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				rho[k][j][i] += rmean[k][j][i];	
			}
		}
	}
    return;
}


void advct(float advx[][j_size][i_size], float v[][j_size][i_size],
		   float u[][j_size][i_size], float dt[][i_size], 
		   float ub[][j_size][i_size], float aam[][j_size][i_size],
		   float vb[][j_size][i_size], float advy[][j_size][i_size]){

//void advct(){

/*change: -advx/-advy
 *
 *
 */
/*
 *advx: read in advance.f
 *
 *v: read and write in advance.f
 *
 *u: read and write in advance.f
 *
 *dt: write again in advance.f dt = h + et
 *
 *ub: read and write in advance.f
 *
 *aam: read and write in advance.f
 *
 *ub: read and write in advance.f
 *
 *advy: read in advance.f
 */

    float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
    float curv[k_size][j_size][i_size];
    float dtaam;

	int i,j,k;

	/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            curv(i,j,k)=0.e0
            advx(i,j,k)=0.e0
            xflux(i,j,k)=0.e0
            yflux(i,j,k)=0.e0
          end do
        end do
      end do
	*/

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				curv[k][j][i] = 0;
				advx[k][j][i] = 0;
				xflux[k][j][i] = 0;
				yflux[k][j][i] = 0;
			}
		}
	}

	/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            curv(i,j,k)=.25e0*((v(i,j+1,k)+v(i,j,k))
     $                         *(dy(i+1,j)-dy(i-1,j))
     $                         -(u(i+1,j,k)+u(i,j,k))
     $                         *(dx(i,j+1)-dx(i,j-1)))
     $                       /(dx(i,j)*dy(i,j))
          end do
        end do
      end do
	*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				curv[k][j][i] = 0.25f*((v[k][j+1][i]+v[k][j][i])
										*(dy[j][i+1]-dy[j][i-1]) 
									  -(u[k][j][i+1]+u[k][j][i])
										*(dx[j+1][i]-dx[j-1][i]))
									 /(dx[j][i]*dy[j][i]);
			}
		}
	}

    //exchange3d_mpi(curv(:,:,1:kbm1),im,jm,kbm1)
    //exchange3d_mpi_xsz_(curv,im,jm,0,kbm1-1);
    //exchange3d_mpi_bak(curv,im,jm,0,kbm1-1);
    exchange3d_mpi(curv,im,jm,kbm1);

//! calculate x-component of velocity advection
//! calculate horizontal advective fluxes

	/*
      do k=1,kbm1
        do j=1,jm
          do i=2,imm1
            xflux(i,j,k)=.125e0*((dt(i+1,j)+dt(i,j))*u(i+1,j,k)
     $                           +(dt(i,j)+dt(i-1,j))*u(i,j,k))
     $                         *(u(i+1,j,k)+u(i,j,k))
          end do
        end do
      end do
	*/

	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 1; i < imm1; i++){
				xflux[k][j][i] = 0.125f*((dt[j][i+1]+dt[j][i])
										  *u[k][j][i+1] 
										+(dt[j][i]+dt[j][i-1])
										  *u[k][j][i])
									   *(u[k][j][i+1]+u[k][j][i]);
			}
		}
	}

	/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            yflux(i,j,k)=.125e0*((dt(i,j)+dt(i,j-1))*v(i,j,k)
     $                           +(dt(i-1,j)+dt(i-1,j-1))*v(i-1,j,k))
     $                         *(u(i,j,k)+u(i,j-1,k))
          end do
        end do
      end do
	*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				yflux[k][j][i] = 0.125f*((dt[j][i]+dt[j-1][i])
										  *v[k][j][i] 
										+(dt[j][i-1]+dt[j-1][i-1])
										  *v[k][j][i-1])
									   *(u[k][j][i]+u[k][j-1][i]);	
			}
		}
	}

//! add horizontal diffusive fluxes
/*
      do k=1,kbm1
        do j=2,jm
          do i=2,imm1
            xflux(i,j,k)=xflux(i,j,k)
     $                    -dt(i,j)*aam(i,j,k)*2.e0
     $                    *(ub(i+1,j,k)-ub(i,j,k))/dx(i,j)
            dtaam=.25e0*(dt(i,j)+dt(i-1,j)+dt(i,j-1)+dt(i-1,j-1))
     $             *(aam(i,j,k)+aam(i-1,j,k)
     $               +aam(i,j-1,k)+aam(i-1,j-1,k))
            yflux(i,j,k)=yflux(i,j,k)
     $                    -dtaam*((ub(i,j,k)-ub(i,j-1,k))
     $                            /(dy(i,j)+dy(i-1,j)
     $                              +dy(i,j-1)+dy(i-1,j-1))
     $                            +(vb(i,j,k)-vb(i-1,j,k))
     $                            /(dx(i,j)+dx(i-1,j)
     $                              +dx(i,j-1)+dx(i-1,j-1)))

            xflux(i,j,k)=dy(i,j)*xflux(i,j,k)
            yflux(i,j,k)=.25e0*(dx(i,j)+dx(i-1,j)
     $                          +dx(i,j-1)+dx(i-1,j-1))*yflux(i,j,k)
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < imm1; i++){
				xflux[k][j][i] = xflux[k][j][i]
								-dt[j][i]*aam[k][j][i]*2.0f
									*(ub[k][j][i+1]-ub[k][j][i])/dx[j][i];
				dtaam = 0.25f*(dt[j][i]+dt[j][i-1]+dt[j-1][i]+dt[j-1][i-1])
							 *(aam[k][j][i]+aam[k][j][i-1]
							  +aam[k][j-1][i]+aam[k][j-1][i-1]);
				yflux[k][j][i] = yflux[k][j][i]
								-dtaam*((ub[k][j][i]-ub[k][j-1][i])
										 /(dy[j][i]+dy[j][i-1]
										  +dy[j-1][i]+dy[j-1][i-1])
									   +(vb[k][j][i]-vb[k][j][i-1])
										 /(dx[j][i]+dx[j][i-1]
										  +dx[j-1][i]+dx[j-1][i-1]));

				xflux[k][j][i] = dy[j][i]*xflux[k][j][i];

				yflux[k][j][i] = 0.25f*(dx[j][i]+dx[j][i-1]
									    +dx[j-1][i]+dx[j-1][i-1])
									  *yflux[k][j][i];

			}
		}
	}

    //exchange3d_mpi(xflux(:,:,1:kbm1),im,jm,kbm1)
    //exchange3d_mpi_xsz_(xflux,im,jm,0,kbm1-1);
    //exchange3d_mpi_bak(xflux,im,jm,0,kbm1-1);
    exchange3d_mpi(xflux,im,jm,kbm1);

//! do horizontal advection
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            advx(i,j,k)=xflux(i,j,k)-xflux(i-1,j,k)
     $                   +yflux(i,j+1,k)-yflux(i,j,k)
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				advx[k][j][i] = xflux[k][j][i]-xflux[k][j][i-1]
							   +yflux[k][j+1][i]-yflux[k][j][i];	
			}
		}
	}

/*
      do k=1,kbm1
        do j=2,jmm1
          if(n_west.eq.-1) then
          do i=3,imm1
            advx(i,j,k)=advx(i,j,k)
     $                   -aru(i,j)*.25e0
     $                     *(curv(i,j,k)*dt(i,j)
     $                        *(v(i,j+1,k)+v(i,j,k))
     $                       +curv(i-1,j,k)*dt(i-1,j)
     $                        *(v(i-1,j+1,k)+v(i-1,j,k)))
          end do
          else
          do i=2,imm1
            advx(i,j,k)=advx(i,j,k)
     $                   -aru(i,j)*.25e0
     $                     *(curv(i,j,k)*dt(i,j)
     $                        *(v(i,j+1,k)+v(i,j,k))
     $                       +curv(i-1,j,k)*dt(i-1,j)
     $                        *(v(i-1,j+1,k)+v(i-1,j,k)))
          end do
          end if
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			if (n_west == -1){
				for (i = 2; i < imm1; i++){
					advx[k][j][i] = advx[k][j][i]
								   -aru[j][i]*0.25f
									*(curv[k][j][i]*dt[j][i]
										*(v[k][j+1][i]+v[k][j][i])
									 +curv[k][j][i-1]*dt[j][i-1]
										*(v[k][j+1][i-1]+v[k][j][i-1]));
				}
			}else{
				for (i = 1; i < imm1; i++){
					advx[k][j][i] = advx[k][j][i]
								   -aru[j][i]*0.25f
									*(curv[k][j][i]*dt[j][i]
										*(v[k][j+1][i]+v[k][j][i])
									 +curv[k][j][i-1]*dt[j][i-1]
										*(v[k][j+1][i-1]+v[k][j][i-1]));
				}
			}
		}
	}

//! calculate y-component of velocity advection

/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            advy(i,j,k)=0.e0
            xflux(i,j,k)=0.e0
            yflux(i,j,k)=0.e0
          end do
        end do
      end do
*/

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				advy[k][j][i] = 0;
				xflux[k][j][i] = 0;
				yflux[k][j][i] = 0;
			}
		}
	}


//! calculate horizontal advective fluxes
/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=.125e0*((dt(i,j)+dt(i-1,j))*u(i,j,k)
     $                           +(dt(i,j-1)+dt(i-1,j-1))*u(i,j-1,k))
     $                         *(v(i,j,k)+v(i-1,j,k))
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i] = 0.125f*((dt[j][i]+dt[j][i-1])
										  *u[k][j][i]
										+(dt[j-1][i]+dt[j-1][i-1])
										  *u[k][j-1][i])
									   *(v[k][j][i]+v[k][j][i-1]);
			}
		}
	}

/*
      do k=1,kbm1
        do j=2,jmm1
          do i=1,im
            yflux(i,j,k)=.125e0*((dt(i,j+1)+dt(i,j))*v(i,j+1,k)
     $                           +(dt(i,j)+dt(i,j-1))*v(i,j,k))
     $                         *(v(i,j+1,k)+v(i,j,k))
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 0; i < im; i++){
				yflux[k][j][i] = 0.125f*((dt[j+1][i]+dt[j][i])*v[k][j+1][i]
										+(dt[j][i]+dt[j-1][i])*v[k][j][i])
									   *(v[k][j+1][i]+v[k][j][i]);
			}
		}
	}

//! add horizontal diffusive fluxes
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,im
            dtaam=.25e0*(dt(i,j)+dt(i-1,j)+dt(i,j-1)+dt(i-1,j-1))
     $             *(aam(i,j,k)+aam(i-1,j,k)
     $               +aam(i,j-1,k)+aam(i-1,j-1,k))
            xflux(i,j,k)=xflux(i,j,k)
     $                    -dtaam*((ub(i,j,k)-ub(i,j-1,k))
     $                            /(dy(i,j)+dy(i-1,j)
     $                              +dy(i,j-1)+dy(i-1,j-1))
     $                            +(vb(i,j,k)-vb(i-1,j,k))
     $                            /(dx(i,j)+dx(i-1,j)
     $                              +dx(i,j-1)+dx(i-1,j-1)))
            yflux(i,j,k)=yflux(i,j,k)
     $                    -dt(i,j)*aam(i,j,k)*2.e0
     $                    *(vb(i,j+1,k)-vb(i,j,k))/dy(i,j)

            xflux(i,j,k)=.25e0*(dy(i,j)+dy(i-1,j)
     $                          +dy(i,j-1)+dy(i-1,j-1))*xflux(i,j,k)
            yflux(i,j,k)=dx(i,j)*yflux(i,j,k)
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < im; i++){
				dtaam = 0.25f*(dt[j][i]+dt[j][i-1]+dt[j-1][i]+dt[j-1][i-1])
							 *(aam[k][j][i]+aam[k][j][i-1]
							   +aam[k][j-1][i]+aam[k][j-1][i-1]);

				xflux[k][j][i] = xflux[k][j][i]
								-dtaam*((ub[k][j][i]-ub[k][j-1][i])
										 /(dy[j][i]+dy[j][i-1]
										  +dy[j-1][i]+dy[j-1][i-1])
									   +(vb[k][j][i]-vb[k][j][i-1])
										 /(dx[j][i]+dx[j][i-1]
										  +dx[j-1][i]+dx[j-1][i-1]));

				yflux[k][j][i] = yflux[k][j][i]
								-dt[j][i]*aam[k][j][i]*2.0f
									*(vb[k][j+1][i]-vb[k][j][i])/dy[j][i];

				xflux[k][j][i] = 0.25f*(dy[j][i]+dy[j][i-1]
									   +dy[j-1][i]+dy[j-1][i-1])
									  *xflux[k][j][i];

				yflux[k][j][i] = dx[j][i]*yflux[k][j][i];
			}
		}
	}

    //  call exchange3d_mpi(yflux(:,:,1:kbm1),im,jm,kbm1)
    //exchange3d_mpi_xsz_(yflux,im,jm,0,kbm1-1);
    //exchange3d_mpi_bak(yflux,im,jm,0,kbm1-1);
    exchange3d_mpi(yflux,im,jm,kbm1);

//! do horizontal advection
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            advy(i,j,k)=xflux(i+1,j,k)-xflux(i,j,k)
     $                   +yflux(i,j,k)-yflux(i,j-1,k)
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				advy[k][j][i] = xflux[k][j][i+1]-xflux[k][j][i]
							   +yflux[k][j][i]-yflux[k][j-1][i];	
			}
		}
	}

/*
      do k=1,kbm1
        do i=2,imm1
          if(n_south.eq.-1) then
          do j=3,jmm1
            advy(i,j,k)=advy(i,j,k)
     $                   +arv(i,j)*.25e0
     $                     *(curv(i,j,k)*dt(i,j)
     $                        *(u(i+1,j,k)+u(i,j,k))
     $                       +curv(i,j-1,k)*dt(i,j-1)
     $                        *(u(i+1,j-1,k)+u(i,j-1,k)))
          end do
          else
          do j=2,jmm1
            advy(i,j,k)=advy(i,j,k)
     $                   +arv(i,j)*.25e0
     $                     *(curv(i,j,k)*dt(i,j)
     $                        *(u(i+1,j,k)+u(i,j,k))
     $                       +curv(i,j-1,k)*dt(i,j-1)
     $                        *(u(i+1,j-1,k)+u(i,j-1,k)))
          end do
          end if
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (i = 1; i < imm1; i++){
			if (n_south == -1){
				for (j = 2; j < jmm1; j++){
					advy[k][j][i] = advy[k][j][i]
								   +arv[j][i]*0.25f
									*(curv[k][j][i]*dt[j][i]
									  *(u[k][j][i+1]+u[k][j][i]) 
									 +curv[k][j-1][i]*dt[j-1][i]
									  *(u[k][j-1][i+1]+u[k][j-1][i]));
				}
			}else{
				for (j = 1; j < jmm1; j++){
					advy[k][j][i] = advy[k][j][i]
								   +arv[j][i]*0.25f
								    *(curv[k][j][i]*dt[j][i]
									  *(u[k][j][i+1]+u[k][j][i]) 
									 +curv[k][j-1][i]*dt[j-1][i]
									  *(u[k][j-1][i+1]+u[k][j-1][i]));	
				}
			}
		}
	}

	return;
}

/*
!_______________________________________________________________________
      subroutine stokes(vorz) !(xstks,ystks,vorz)
!
! given u,vstks (from wind.f) this routine calculates the Stokes forces
!     both vortex and coriolis x,ystks in m/s^2; these are then 
!     to be saved in SRFS.*.nc file.  In that file, the 1-->kb/2 contain 
!     the x,ystks, but kb/2+1-->kbm1 has u,vstks
!
!     Note: ustks, vstks & vorz all defined at corner of u(i,j) & v(i,j)
!
!     The routine 1st computes
!     xstks = [Stokes (vortex + coriolis) acc]*cell_area*depth @ U-pnt
!     ystks = [Stokes (vortex + coriolis) acc]*cell_area*depth @ V-pnt
!     then:
!     x,ystks are added to advx,y to be used in advu,v & mode_interaction
!     finally:
!     x,ystks are reversed & divided by cell_area*depth to give m/s^2
!
      implicit none
      include 'pom.h'
      real vorz(im,jm,kb) !,xstks(im,jm,kb),ystks(im,jm,kb)
      integer i,j,k
*/

void stokes(){

/*
      vorz(:,:,:)=0.e0; xstks(:,:,:)=0.e0; ystks(:,:,:)=0.e0

! sigma-level related z_vorticity term (*corner_cell_area*depth)
      k=1
        do j=2,jm
          do i=2,im
            vorz(i,j,k)=
     $     -0.125*(dy(i,j)+dy(i,j-1)+dy(i-1,j)+dy(i-1,j-1))
     $           *((v(i,j,k  )+v(i-1,j,k  )-v(i,j,k+1)-v(i-1,j,k+1))
     $                     /(zz(k  )-zz(k+1)))
     $            *0.5*(zz(k)*(dt(i,j)+dt(i,j-1)-dt(i-1,j)-dt(i-1,j-1))
     $                       +(et(i,j)+et(i,j-1)-et(i-1,j)-et(i-1,j-1)))
     $     +0.125*(dx(i,j)+dx(i,j-1)+dx(i-1,j)+dx(i-1,j-1))
     $           *((u(i,j,k  )+u(i,j-1,k  )-u(i,j,k+1)-u(i,j-1,k+1))
     $                     /(zz(k  )-zz(k+1)))
     $            *0.5*(zz(k)*(dt(i,j)+dt(i-1,j)-dt(i,j-1)-dt(i-1,j-1))
     $                       +(et(i,j)+et(i-1,j)-et(i,j-1)-et(i-1,j-1)))
          end do
        end do
      do k=2,kbm2
        do j=2,jm
          do i=2,im
            vorz(i,j,k)=
     $     -0.125*(dy(i,j)+dy(i,j-1)+dy(i-1,j)+dy(i-1,j-1))
     $           *((v(i,j,k-1)+v(i-1,j,k-1)-v(i,j,k+1)-v(i-1,j,k+1))
     $                     /(zz(k-1)-zz(k+1)))
     $            *0.5*(zz(k)*(dt(i,j)+dt(i,j-1)-dt(i-1,j)-dt(i-1,j-1))
     $                       +(et(i,j)+et(i,j-1)-et(i-1,j)-et(i-1,j-1)))
     $     +0.125*(dx(i,j)+dx(i,j-1)+dx(i-1,j)+dx(i-1,j-1))
     $           *((u(i,j,k-1)+u(i,j-1,k-1)-u(i,j,k+1)-u(i,j-1,k+1))
     $                     /(zz(k-1)-zz(k+1)))
     $            *0.5*(zz(k)*(dt(i,j)+dt(i-1,j)-dt(i,j-1)-dt(i-1,j-1))
     $                       +(et(i,j)+et(i-1,j)-et(i,j-1)-et(i-1,j-1)))
          end do
        end do
      end do
      k=kbm1
        do j=2,jm
          do i=2,im
            vorz(i,j,k)=
     $     -0.125*(dy(i,j)+dy(i,j-1)+dy(i-1,j)+dy(i-1,j-1))
     $           *((v(i,j,k-1)+v(i-1,j,k-1)-v(i,j,k  )-v(i-1,j,k  ))
     $                     /(zz(k-1)-zz(k  )))
     $            *0.5*(zz(k)*(dt(i,j)+dt(i,j-1)-dt(i-1,j)-dt(i-1,j-1))
     $                       +(et(i,j)+et(i,j-1)-et(i-1,j)-et(i-1,j-1)))
     $     +0.125*(dx(i,j)+dx(i,j-1)+dx(i-1,j)+dx(i-1,j-1))
     $           *((u(i,j,k-1)+u(i,j-1,k-1)-u(i,j,k  )-u(i,j-1,k  ))
     $                     /(zz(k-1)-zz(k  )))
     $            *0.5*(zz(k)*(dt(i,j)+dt(i-1,j)-dt(i,j-1)-dt(i-1,j-1))
     $                       +(et(i,j)+et(i-1,j)-et(i,j-1)-et(i-1,j-1)))
          end do
        end do

      do k=1,kbm1    !add z_vorticity*corner_cell_area*depth
        do j=2,jm    
          do i=2,im
            vorz(i,j,k)=vorz(i,j,k)+
     $      0.125*(dt(i,j)+dt(i,j-1)+dt(i-1,j)+dt(i-1,j-1))
     $           *( v(i,j,k)  *(dy(i,j)  +dy(i,j-1)  )
     $             -v(i-1,j,k)*(dy(i-1,j)+dy(i-1,j-1))
     $             -u(i,j,k)  *(dx(i,j)  +dx(i-1,j)  )
     $             +u(i,j-1,k)*(dx(i,j-1)+dx(i-1,j-1)) )
          end do
        end do
      end do

        do j=2,jm    !add coriolis*corner_cell_area*depth
          do i=2,im
            vorz(i,j,kb)=      !Use (:,:,kb) for dummy
     $      0.03125*(dt(i,j)+dt(i,j-1)+dt(i-1,j)+dt(i-1,j-1))
     $             *(aru(i,j)+aru(i,j-1))
     $             *(cor(i,j)+cor(i,j-1)+cor(i-1,j)+cor(i-1,j-1))
          end do
        end do
        do k=1,kbm1
           vorz(:,:,k)=vorz(:,:,k)+vorz(:,:,kb)
           end do
!
      vorz(1,:,:)=vorz(2,:,:); vorz(:,1,:)=vorz(:,2,:) !end points
!
      do k=1,kbm1
        do j=1,jm
          do i=1,im
             xstks(i,j,k)=-vstks(i,j,k)*vorz(i,j,k) !x-Vortex+CorForce LHS
             ystks(i,j,k)= ustks(i,j,k)*vorz(i,j,k) !y-Vortex+CorForce LHS
          end do
        end do
      end do

      call exchange3d_mpi(xstks(:,:,1:kbm1),im,jm,kbm1)
      call exchange3d_mpi(ystks(:,:,1:kbm1),im,jm,kbm1)

      do k=1,kbm1
        do j=1,jmm1
          do i=1,im
             xstks(i,j,k)=0.5*(xstks(i,j,k)+xstks(i,j+1,k)) !U-point
          end do
        end do
        do j=1,jm
          do i=1,imm1
             ystks(i,j,k)=0.5*(ystks(i,j,k)+ystks(i+1,j,k)) !V-point
          end do
        end do
      end do

      xstks(:,jm,:)=xstks(:,jmm1,:); ystks(im,:,:)=ystks(imm1,:,:)

      do k=1,kbm1
        do j=2,jm
          do i=2,im
             advx(i,j,k)=advx(i,j,k)+xstks(i,j,k)
             advy(i,j,k)=advy(i,j,k)+ystks(i,j,k)
          end do
        end do
      end do

! divide by (cell_area*depth) & reverse signs --> Stokes forces on RHS:
! note: xstks = vstks*(f+zeta) & ystks =-ustks*(f+zeta)
!     where zeta = z_vorticity; so Stokes forces can be separated
!     if x,ystks and u,vstks are saved
      do k=1,kbm1
        do j=1,jm
          do i=2,im
             xstks(i,j,k)=-xstks(i,j,k)/((dt(i,j)+dt(i-1,j))*aru(i,j))
          end do
        end do
        do j=2,jm
          do i=1,im
             ystks(i,j,k)=-ystks(i,j,k)/((dt(i,j)+dt(i,j-1))*arv(i,j))
          end do
        end do
      end do
      xstks(1,:,:)=xstks(2,:,:);    ystks(:,1,:)=ystks(:,2,:);

      return
      end
*/
}


/*
void advave(float advua[][i_size], float d[][i_size],
		    float ua[][i_size], float va[][i_size],
			float uab[][i_size], float aam2d[][i_size],
			float vab[][i_size], float advva[][i_size],
			float wubot[][i_size], float wvbot[][i_size]){
*/

void advave(){

	int i,j,k;
	float tps[j_size][i_size];
    float curv2d[j_size][i_size];
    float fluxua[j_size][i_size];
    float fluxva[j_size][i_size];


/*change: 
 *   need not copy in: advua/fluxua/fluxva/advva     wubot/wvbot/aam2d
 *   need copy in : none
 *
 *reference:
 *   others
 *
 */
/*
 * advua: used in advance.f (read)
 *        write in restart file
 *        in this function, initialize as 0 and then calculate and export
 *
 * d: changed in advance.f d = h + el 
 * 
 * ua: read and write in advance.f
 *
 * va: read and write in advance.f
 *
 * fluxua: read and write in advance.f(indeed a tmp array)
 *
 * fluxva: read and write in advance.f(indeed a tmp array)
 *
 * uab: read and write in advance.f
 *
 *
 * vab: read and write in advance.f
 * 
 * tps : read and write in advance.f
 *       but in advance.f tps is always assigned a value before
 *       so I believe it is a tmp array; do not reference in
 *
 * wubot: read in advance.f and assigned in this function
 *
 * wvbot: read in advance.f and assigned in this function
 */
/*
      subroutine advave
! calculate horizontal advection and diffusion
      implicit none
      include 'pom.h'
      real curv2d(im,jm)
      integer i,j

      double precision advave_time_start_xsz
      double precision advave_time_end_xsz

      call call_time_start(advave_time_start_xsz)

! u-advection and diffusion
*/

/*
! advective fluxes
      do j=1,jm
        do i=1,im
          advua(i,j)=0.e0
        end do
      end do
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			advua[j][i] = 0;	// for read and write restart file
		}
	}
/*
      do j=2,jm
        do i=2,imm1
          fluxua(i,j)=.125e0*((d(i+1,j)+d(i,j))*ua(i+1,j)
     $                       +(d(i,j)+d(i-1,j))*ua(i,j))
     $                      *(ua(i+1,j)+ua(i,j))
        end do
      end do
*/
	for (j = 1; j < jm; j++){
		for (i = 1; i < imm1; i++){
			fluxua[j][i] = 0.125f*((d[j][i+1]+d[j][i])*ua[j][i+1]+(d[j][i]+d[j][i-1])*ua[j][i])*(ua[j][i+1]+ua[j][i]);
		}
	}

/*
      do j=2,jm
        do i=2,im
          fluxva(i,j)=.125e0*((d(i,j)+d(i,j-1))*va(i,j)
     $                       +(d(i-1,j)+d(i-1,j-1))*va(i-1,j))
     $                      *(ua(i,j)+ua(i,j-1))
        end do
      end do
*/
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = 0.125f*((d[j][i]+d[j-1][i])*va[j][i] + (d[j][i-1]+d[j-1][i-1])*va[j][i-1])*(ua[j][i] + ua[j-1][i]);
		}
	}
/*
! add viscous fluxes
      do j=2,jm
        do i=2,imm1
          fluxua(i,j)=fluxua(i,j)
     $                 -d(i,j)*2.e0*aam2d(i,j)*(uab(i+1,j)-uab(i,j))
     $                   /dx(i,j)
        end do
      end do
*/
	for (j = 1; j < jm; j++){
		for (i = 1; i < imm1; i++){
			fluxua[j][i] = fluxua[j][i] - d[j][i]*2.0f*aam2d[j][i]*(uab[j][i+1]-uab[j][i])/dx[j][i];
		}
	}
/*
      do j=2,jm
        do i=2,im
          tps(i,j)=.25e0*(d(i,j)+d(i-1,j)+d(i,j-1)+d(i-1,j-1))
     $              *(aam2d(i,j)+aam2d(i,j-1)
     $                +aam2d(i-1,j)+aam2d(i-1,j-1))
     $              *((uab(i,j)-uab(i,j-1))
     $                 /(dy(i,j)+dy(i-1,j)+dy(i,j-1)+dy(i-1,j-1))
     $               +(vab(i,j)-vab(i-1,j))
     $                 /(dx(i,j)+dx(i-1,j)+dx(i,j-1)+dx(i-1,j-1)))
          fluxua(i,j)=fluxua(i,j)*dy(i,j)
          fluxva(i,j)=(fluxva(i,j)-tps(i,j))*.25e0
     $                 *(dx(i,j)+dx(i-1,j)+dx(i,j-1)+dx(i-1,j-1))
        end do
      end do
*/
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			tps[j][i] = 0.25f*(d[j][i]+d[j][i-1]+d[j-1][i]+d[j-1][i-1])*(aam2d[j][i]+aam2d[j-1][i]+aam2d[j][i-1]+aam2d[j-1][i-1])*((uab[j][i]-uab[j-1][i])/(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1])+(vab[j][i]-vab[j][i-1])/(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1]));
			fluxua[j][i] *= dy[j][i];
			fluxva[j][i] = (fluxva[j][i]-tps[j][i])*0.25f*(dx[j][i]+dx[j][i-1]+dx[j-1][i]+dx[j-1][i-1]);
		}
	}

    //exchange2d_mpi_xsz_(fluxua,im,jm);
    exchange2d_mpi(fluxua,im,jm);
/*
      do j=2,jmm1
        do i=2,imm1
          advua(i,j)=fluxua(i,j)-fluxua(i-1,j)
     $                +fluxva(i,j+1)-fluxva(i,j)
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			advua[j][i] = fluxua[j][i]-fluxua[j][i-1]+fluxva[j+1][i]-fluxva[j][i];
		}
	}

/*
! v-advection and diffusion
      do j=1,jm
        do i=1,im
          advva(i,j)=0.e0
        end do
      end do
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			advva[j][i] = 0;	
		}
	}

/*
! advective fluxes
      do j=2,jm
        do i=2,im
          fluxua(i,j)=.125e0*((d(i,j)+d(i-1,j))*ua(i,j)
     $                       +(d(i,j-1)+d(i-1,j-1))*ua(i,j-1))
     $                      *(va(i-1,j)+va(i,j))
        end do
      end do
*/
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			fluxua[j][i] = 0.125f*((d[j][i]+d[j][i-1])*ua[j][i]+(d[j-1][i]+d[j-1][i-1])*ua[j-1][i])*(va[j][i-1]+va[j][i]);
		}
	}

/*
      do j=2,jmm1
        do i=2,im
          fluxva(i,j)=.125e0*((d(i,j+1)+d(i,j))*va(i,j+1)
     $                       +(d(i,j)+d(i,j-1))*va(i,j))
     $                      *(va(i,j+1)+va(i,j))
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = 0.125f*((d[j+1][i]+d[j][i])*va[j+1][i] + (d[j][i]+d[j-1][i])*va[j][i])*(va[j+1][i]+va[j][i]);
		}
	}

/*
! add viscous fluxes
      do j=2,jmm1
        do i=2,im
          fluxva(i,j)=fluxva(i,j)
     $                 -d(i,j)*2.e0*aam2d(i,j)*(vab(i,j+1)-vab(i,j))
     $                   /dy(i,j)
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = fluxva[j][i] - d[j][i]*2.0f*aam2d[j][i]*(vab[j+1][i]-vab[j][i])/dy[j][i];	
		}
	}

/*
      do j=2,jm
        do i=2,im
          fluxva(i,j)=fluxva(i,j)*dx(i,j)
          fluxua(i,j)=(fluxua(i,j)-tps(i,j))*.25e0
     $                 *(dy(i,j)+dy(i-1,j)+dy(i,j-1)+dy(i-1,j-1))
        end do
      end do
*/
	for (j = 1; j < jm; j++){
		for (i = 1; i < im; i++){
			fluxva[j][i] = fluxva[j][i]*dx[j][i];	
			fluxua[j][i] = (fluxua[j][i]-tps[j][i])*0.25f*(dy[j][i]+dy[j][i-1]+dy[j-1][i]+dy[j-1][i-1]);
		}
	}
    //exchange2d_mpi_xsz_(fluxva,im,jm);
    exchange2d_mpi(fluxva,im,jm);
/*
      do j=2,jmm1
        do i=2,imm1
          advva(i,j)=fluxua(i+1,j)-fluxua(i,j)
     $                +fluxva(i,j)-fluxva(i,j-1)
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			advva[j][i] = fluxua[j][i+1]-fluxua[j][i]+fluxva[j][i]-fluxva[j-1][i];	
		}
	}

/*
      if(mode.eq.2) then
*/

	if (mode == 2){
		/*
        do j=2,jmm1
          do i=2,imm1
            wubot(i,j)=-0.5e0*(cbc(i,j)+cbc(i-1,j))
     $                  *sqrt(uab(i,j)**2
     $                        +(.25e0*(vab(i,j)+vab(i,j+1)
     $                                 +vab(i-1,j)+vab(i-1,j+1)))**2)
     $                  *uab(i,j)
          end do
        end do
		*/
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				float tmp = 0.25f*(vab[j][i]+vab[j+1][i]
								  +vab[j][i-1]+vab[j+1][i-1]);
				wubot[j][i] = -0.5f*(cbc[j][i]+cbc[j][i-1])
								   *sqrtf(uab[j][i]*uab[j][i]+tmp*tmp)
								   *uab[j][i];
			}
		}

        exchange2d_mpi(wubot,im,jm);

		/*
        do j=2,jmm1
          do i=2,imm1
            wvbot(i,j)=-0.5e0*(cbc(i,j)+cbc(i,j-1))
     $                  *sqrt(vab(i,j)**2
     $                        +(.25e0*(uab(i,j)+uab(i+1,j)
     $                                +uab(i,j-1)+uab(i+1,j-1)))**2)
     $                  *vab(i,j)
          end do
        end do
		*/
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				float tmp = 0.25f*(uab[j][i]+uab[j][i+1]
								  +uab[j-1][i]+uab[j-1][i+1]);
				wvbot[j][i] = -0.5f*(cbc[j][i]+cbc[j-1][i])
								   *sqrtf(vab[j][i]*vab[j][i]+tmp*tmp)
								   *vab[j][i];
				//wvbot[j][i] = -0.5f*(cbc[j][i]+cbc[j-1][i])*sqrtf(powf(vab[j][i],2)+powf(tmp,2))*vab[j][i];
				//printf("I come here!\n");
			}
		}

        exchange2d_mpi(wvbot,im,jm);

		/*
        do j=2,jmm1
          do i=2,imm1
            curv2d(i,j)=.25e0
     $                   *((va(i,j+1)+va(i,j))*(dy(i+1,j)-dy(i-1,j))
     $                    -(ua(i+1,j)+ua(i,j))*(dx(i,j+1)-dx(i,j-1)))
     $                   /(dx(i,j)*dy(i,j))
          end do
        end do
		*/
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				curv2d[j][i] = 0.25f*((va[j+1][i]+va[j][i])
									  *(dy[j][i+1]-dy[j][i-1])
									 -(ua[j][i+1]+ua[j][i])
									  *(dx[j+1][i]-dx[j-1][i]))
								/(dx[j][i]*dy[j][i]);	//xsz
			}
		}
        //exchange2d_mpi_xsz_(curv2d,im,jm);
        exchange2d_mpi(curv2d,im,jm);

		/*
        do j=2,jmm1
          if(n_west.eq.-1) then
          do i=3,imm1
            advua(i,j)=advua(i,j)-aru(i,j)*.25e0
     $                  *(curv2d(i,j)*d(i,j)
     $                    *(va(i,j+1)+va(i,j))
     $                    +curv2d(i-1,j)*d(i-1,j)
     $                    *(va(i-1,j+1)+va(i-1,j)))
          end do
          else
          do i=2,imm1
            advua(i,j)=advua(i,j)-aru(i,j)*.25e0
     $                  *(curv2d(i,j)*d(i,j)
     $                    *(va(i,j+1)+va(i,j))
     $                    +curv2d(i-1,j)*d(i-1,j)
     $                    *(va(i-1,j+1)+va(i-1,j)))
          end do
          end if
        end do
		*/
		for (j = 1; j < jmm1; j++){
			if (n_west == -1){
				for (i = 2; i < imm1; i++){
					advua[j][i] = advua[j][i]
								  -aru[j][i]*0.25f
									*(curv2d[j][i]*d[j][i]
									   *(va[j+1][i]+va[j][i])
									 +curv2d[j][i-1]*d[j][i-1]
									   *(va[j+1][i-1]+va[j][i-1]));
				}
			}else{
				for (i = 1; i < imm1; i++){
					advua[j][i] = advua[j][i]
						          -aru[j][i]*0.25f
								    *(curv2d[j][i]*d[j][i]
									   *(va[j+1][i]+va[j][i])
									 +curv2d[j][i-1]*d[j][i-1]
									   *(va[j+1][i-1]+va[j][i-1]));
				}
			}
		}

		/*
        do i=2,imm1
          if(n_south.eq.-1) then
          do j=3,jmm1
            advva(i,j)=advva(i,j)+arv(i,j)*.25e0
     $                  *(curv2d(i,j)*d(i,j)
     $                    *(ua(i+1,j)+ua(i,j))
     $                    +curv2d(i,j-1)*d(i,j-1)
     $                    *(ua(i+1,j-1)+ua(i,j-1)))
          end do
          else
          do j=2,jmm1
            advva(i,j)=advva(i,j)+arv(i,j)*.25e0
     $                  *(curv2d(i,j)*d(i,j)
     $                    *(ua(i+1,j)+ua(i,j))
     $                    +curv2d(i,j-1)*d(i,j-1)
     $                    *(ua(i+1,j-1)+ua(i,j-1)))
          end do
          end if
        end do
		*/
		for (i = 1; i < imm1; i++){
			if (n_south == -1){
				for (j = 2; j < jmm1; j++){
					advva[j][i] = advva[j][i]
						         +arv[j][i]*0.25f
									*(curv2d[j][i]*d[j][i]
										*(ua[j][i+1]+ua[j][i])
									 +curv2d[j-1][i]*d[j-1][i]
										*(ua[j-1][i+1]+ua[j-1][i]));
				}
			}else{
				for (j = 1; j < jmm1; j++){
					advva[j][i] = advva[j][i]
								 +arv[j][i]*0.25f
									*(curv2d[j][i]*d[j][i]
										*(ua[j][i+1]+ua[j][i])
									 +curv2d[j-1][i]*d[j-1][i]
										*(ua[j-1][i+1]+ua[j-1][i]));	
				}
			}
		}
		/*
!lyo:channel:
          do j=2,jmm1
            do i=2,imm1
!!          aam2d(i,j)=aam_init*(1.+aamfrz(i,j)) !lyo:channel:
            aam2d(i,j)=(horcon*dx(i,j)*dy(i,j)
!           aam2d(i,j)=horcon*dx(i,j)*dy(i,j)*(1.+aamfrz(i,j)) !lyo:channel:
     $                    *sqrt( ((ua(i+1,j)-ua(i,j))/dx(i,j))**2
     $                          +((va(i,j+1)-va(i,j))/dy(i,j))**2
     $                    +.5e0*(.25e0*(ua(i,j+1)+ua(i+1,j+1)
     $                                 -ua(i,j-1)-ua(i+1,j-1))
     $                    /dy(i,j)
     $                    +.25e0*(va(i+1,j)+va(i+1,j+1)
     $                           -va(i-1,j)-va(i-1,j+1))
     $                    /dx(i,j)) **2)
     $              +aam_init)*(1.+aamfrz(i,j)) !lyo:channel:
!!   $              + 500.0
            end do
          end do
		*/
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				float tmpu = (ua[j][i+1]-ua[j][i])/dx[j][i];
				float tmpv = (va[j+1][i]-va[j][i])/dy[j][i];
				float tmpuv = 0.25f*(ua[j+1][i]+ua[j+1][i+1]
									 -ua[j-1][i]-ua[j-1][i+1])/dy[j][i]
						     +0.25f*(va[j][i+1]+va[j+1][i+1]
									 -va[j][i-1]-va[j+1][i-1])/dx[j][i];

				aam2d[j][i] = (horcon*dx[j][i]*dy[j][i]
								*sqrtf((tmpu*tmpu)+(tmpv*tmpv)
									   +0.5f*(tmpuv*tmpuv))
							   +aam_init)
							  *(1.f+aamfrz[j][i]);		//!lyo:channel:
			}
		}
	}
/*
      end if
*/
	return;
}


/*
      SUBROUTINE VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CF)
C
      implicit none
c
      integer, intent(in) :: IM,JM
      real, dimension(im,jm), intent(in) :: FX,FY,DX,DY,DUM,DVM
      real, dimension(im,jm), intent(out) :: CF
      real  C,AREA
      integer i,j
!
!lyo:vort:necessary to use 2,im & 2,jm to match domains in parallel
*/
void vort_curl(float fx[][i_size], float fy[][i_size],
			   float dx[][i_size], float dy[][i_size],
			   float dum[][i_size], float dvm[][i_size],
			   int im, int jm,
			   float cf[][i_size]){
/*
!     DO I=3,IM-1
!     DO J=3,JM-1
      DO I=2,IM
      DO J=2,JM
C
      C=-FX(I,J)*(DX(I,J)+DX(I-1,J))+FX(I,J-1)*(DX(I,J-1)+DX(I-1,J-1))
     2  +FY(I,J)*(DY(I,J)+DY(I,J-1))-FY(I-1,J)*(DY(I-1,J)+DY(I-1,J-1))
      CF(I,J)=   C*DUM(I,J)*DUM(I,J-1)*DVM(I,J)*DVM(I-1,J)
C
      AREA   =   0.25*(DX(I,J)+DX(I-1,J) +DX(I,J-1)+DX(I-1,J-1))
     2          *0.25*(DY(I,J)+DY(I,J-1) +DY(I-1,J)+DY(I-1,J-1))
      CF(I,J)=CF(I,J)/AREA
C
      ENDDO
      ENDDO
      RETURN
      END
*/
	int i, j;
	float c, area;

	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			c = -fx[j][i]*(dx[j][i]+dx[j][i-1])
				+fx[j-1][i]*(dx[j-1][i]+dx[j-1][i-1])
				+fy[j][i]*(dy[j][i]+dy[j-1][i])
				-fy[j][i-1]*(dy[j][i-1]+dy[j-1][i-1]);

			cf[j][i] = c*dum[j][i]*dum[j-1][i]
					  *dvm[j][i]*dvm[j][i-1];

			area = 0.25f*(dx[j][i]+dx[j][i-1]
						 +dx[j-1][i]+dx[j-1][i-1])
				  *0.25f*(dy[j][i]+dy[j-1][i]
						 +dy[j][i-1]+dy[j-1][i-1]);

			cf[j][i] = cf[j][i]/area;
		}
	}
}

/*
!_______________________________________________________________________
!
!lyo:!vort:beg:Vorticity(e.g. JEBAR) analysis, from:
!/wrk/aden/lyo/pom_gfdex/wmo09training/anIntroCourseNumOceanExpsUsingPOM
!     pom08.f_master & pom08.c
!_______________________________________________________________________
      SUBROUTINE VORT !*(nhra,nu) !(ADVUA,ADVVA,ADX2D,ADY2D,DRX2D,DRY2D,IIM,JJM)
C **********************************************************************
C *                                                                    *
C * FUNCTION    :  Vorticity analysis                                  *
C *                                                                    *
C **********************************************************************
C
      implicit none
      include 'pom.h'
!     Local variables:
      integer i,j,IID !*,nhra,nu
      real dmx,dmy
C
C --- IID=1 divide terms by D ; IID=0 no div. by D
      IID=1 !=1 results in curl of depth-averaged   momentum eqns
	    !=0 results in curl of depth-integrated momentum eqns
C
*/
void vort(){
	int i, j, iid;
	float dmx, dmy;
	iid = 1;

/*
      do j=1,jm
      do i=1,im
      FX(i,j)=0.
      FY(i,j)=0.
      CTSURF(i,j)=0.
      CTBOT(i,j)=0.
      CPVF(i,j)=0.
      CJBAR(i,j)=0.
      CADV(i,j)=0.
      CTEN(i,j)=0.
      CELG(i,j)=0.
      CTOT(i,j)=0.
      TOTX(i,j)=0.
      TOTY(i,j)=0.
      enddo
      enddo
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			fx[j][i] = 0;
			fy[j][i] = 0;
			ctsurf[j][i] = 0;
			ctbot[j][i] = 0;
			cpvf[j][i] = 0;
			cjbar[j][i] = 0;
			cadv[j][i] = 0;
			cten[j][i] = 0;
			celg[j][i] = 0;
			ctot[j][i] = 0;
			totx[j][i] = 0;
			toty[j][i] = 0;
		}
	}
/*
C
C --------------- surface stress term ------------------------
!     Note wu,vsurf are defined at (u,v)-points
      DO 10 I=2,IM
      DO 10 J=2,JM
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=WUSURF(I,J)/DMX
      FY(I,J)=WVSURF(I,J)/DMY
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  10  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CTSURF)
      call exchange2d_mpi( ctsurf, im, jm )
*/
	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.0f;
				dmy = 1.0f;
			}

			fx[j][i] = wusurf[j][i]/dmx;
			fy[j][i] = wvsurf[j][i]/dmy;
			totx[j][i] = totx[j][i] + fx[j][i];
			toty[j][i] = toty[j][i] + fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);

	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, ctsurf);
	exchange2d_mpi(ctsurf, im, jm);

/*
C
C --------------- bottom  stress term ------------------------
!     Note wu,vbot are defined at (u,v)-points
      DO 20 I=2,IM
      DO 20 J=2,JM
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=-WUBOT(I,J)/DMX
      FY(I,J)=-WVBOT(I,J)/DMY
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  20  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CTBOT)
      call exchange2d_mpi( ctbot,  im, jm )
*/
	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.f;
				dmy = 1.f;
			}
			fx[j][i] = -wubot[j][i]/dmx;
			fy[j][i] = -wvbot[j][i]/dmy;
			totx[j][i] = totx[j][i]+fx[j][i];
			toty[j][i] = toty[j][i]+fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);

	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, ctbot);
	exchange2d_mpi(ctbot, im, jm);

/*
C
C ------------------ surf. elev. gradient term ---------------
!     CELG is in general small
!     ALPHA =0.225
C
      DO 30 I=2,IM
      DO 30 J=2,JM
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=.25*GRAV*(DY(I,J)+DY(I-1,J))*(D(I,J)+D(I-1,J))
     4             *( (1.-2.*ALPHA)*(EL(I,J)-EL(I-1,J))
     4            +ALPHA*(ELB(I,J)-ELB(I-1,J)+ELF(I,J)-ELF(I-1,J)) )
      FX(I,J)=FX(I,J)/(ARU(I,J)*DMX)
C
      FY(I,J)=.250*GRAV*(DX(I,J)+DX(I,J-1))*(D(I,J)+D(I,J-1))
     4                   *( (1.0-2.0*ALPHA)*(EL(I,J)-EL(I,J-1))
     4            +ALPHA*(ELB(I,J)-ELB(I,J-1)+ELF(I,J)-ELF(I,J-1)) )
      FY(I,J)=FY(I,J)/(ARV(I,J)*DMY)
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  30  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CELG)
      call exchange2d_mpi( celg,   im, jm )
*/
	alpha = 0.225f;
	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);	
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.f;
				dmy = 1.f;
			}
			fx[j][i] = 0.25f*grav*(dy[j][i]+dy[j][i-1])
					  *(d[j][i]+d[j][i-1])
					  *((1.f-2.f*alpha)*(el[j][i]-el[j][i-1])
					    +alpha*(elb[j][i]-elb[j][i-1]
							   +elf[j][i]-elf[j][i-1]));

			fx[j][i] = fx[j][i]/(aru[j][i]*dmx);

			fy[j][i] = 0.25f*grav*(dx[j][i]+dx[j-1][i])
					  *(d[j][i]+d[j-1][i])
					  *((1.f-2.f*alpha)*(el[j][i]-el[j-1][i])
					    +alpha*(elb[j][i]-elb[j-1][i]
						       +elf[j][i]-elf[j-1][i]));

			fy[j][i] = fy[j][i]/(arv[j][i]*dmy);
			totx[j][i] = totx[j][i] + fx[j][i];
			toty[j][i] = toty[j][i] + fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);
	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, celg);
	exchange2d_mpi(celg, im, jm);

/*
C
C -------------------- JBAR term (incl. elev grad)------------
!     For IID=1:
!     CJBAR =~ -Jacobian(Kai,1/H); where Kai=Integral_H_0{zbdz}
      DO 40 I=2,IM
      DO 40 J=2,JM
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=DRX2D(I,J)/(ARU(I,J)*DMX)
      FY(I,J)=DRY2D(I,J)/(ARV(I,J)*DMY)
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  40  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CJBAR)
      call exchange2d_mpi( cjbar,  im, jm )
C
      DO J=1,JM
      DO I=1,IM
      CJBAR(I,J)=CJBAR(I,J)+CELG(I,J)
      ENDDO
      ENDDO
*/

	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.f;
				dmy = 1.f;
			}

			fx[j][i] = drx2d[j][i]/(aru[j][i]*dmx);
			fy[j][i] = dry2d[j][i]/(arv[j][i]*dmy);

			totx[j][i] = totx[j][i]+fx[j][i];
			toty[j][i] = toty[j][i]+fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);
	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, cjbar);
	exchange2d_mpi(cjbar, im, jm);

/*
C
C --------------- advection and diffusion terms --------------
      DO 50 I=2,IM
      DO 50 J=2,JM
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=(ADX2D(I,J)+ADVUA(I,J))/(ARU(I,J)*DMX)
      FY(I,J)=(ADY2D(I,J)+ADVVA(I,J))/(ARV(I,J)*DMY)
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  50  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CADV)
      call exchange2d_mpi( cadv,   im, jm )
*/

	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.f;
				dmy = 1.f;
			}
			fx[j][i] = (adx2d[j][i]+advua[j][i])/(aru[j][i]*dmx);
			fy[j][i] = (ady2d[j][i]+advva[j][i])/(arv[j][i]*dmy);
			totx[j][i] = totx[j][i]+fx[j][i];
			toty[j][i] = toty[j][i]+fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);
	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, cadv);
	exchange2d_mpi(cadv, im, jm);

/*
C
C ---------------------- coriolis term -----------------------
!     For IID=1:
!     CPVF =~ [H.(UA,VA)]dot GRAD[f/H]
      DO 60 I=2,IMM1  !"arr-bounds-exceed" !Ying-TsaoLee(WMO09class)
      DO 60 J=2,JMM1
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=-.25*(  COR(I,J)*D(I,J)*(VA(I,J+1)+VA(I,J))
     2        +COR(I-1,J)*D(I-1,J)*(VA(I-1,J+1)+VA(I-1,J)))/DMX
      FY(I,J)=+.25*(  COR(I,J)*D(I,J)*(UA(I+1,J)+UA(I,J))
     2        +COR(I,J-1)*D(I,J-1)*(UA(I+1,J-1)+UA(I,J-1)))/DMY
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  60  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CPVF)
      call exchange2d_mpi( cpvf,   im, jm )
*/

	for (i = 1; i < imm1; i++){
		for (j = 1; j < jmm1; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.f;
				dmy = 1.f;
			}

			fx[j][i] = -0.25f*(cor[j][i]*d[j][i]
								*(va[j+1][i]+va[j][i])
							  +cor[j][i-1]*d[j][i-1]
							    *(va[j+1][i-1]+va[j][i-1]))/dmx;
			fy[j][i] = 0.25f*(cor[j][i]*d[j][i]
								*(ua[j][i+1]+ua[j][i])
							 +cor[j-1][i]*d[j-1][i]
								*(ua[j-1][i+1]+ua[j-1][i]))/dmy;

			totx[j][i] = totx[j][i]+fx[j][i];
			toty[j][i] = toty[j][i]+fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);
	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, cpvf);
	exchange2d_mpi(cpvf, im, jm);

/*
C
C ------------------ tendency (d/dt) term --------------------
      DO 70 I=2,IM
      DO 70 J=2,JM
       IF(IID.EQ.1) THEN
      DMX=0.5*(D(I,J)+D(I-1,J))
      DMY=0.5*(D(I,J)+D(I,J-1))
       ELSE
      DMX=1.
      DMY=1.
       ENDIF
      FX(I,J)=(UAF(I,J)*(H(I,J)+ELF(I,J)+H(I-1,J)+ELF(I-1,J))
     1  -UAB(I,J)*(H(I,J)+ELB(I,J)+H(I-1,J)+ELB(I-1,J)))/
     2  (4.*DTE*DMX)*DUM(I,J)
      FY(I,J)=(VAF(I,J)*(H(I,J)+ELF(I,J)+H(I,J-1)+ELF(I,J-1))
     1  -VAB(I,J)*(H(I,J)+ELB(I,J)+H(I,J-1)+ELB(I,J-1)))/
     2  (4.*DTE*DMY)*DVM(I,J)
      TOTX(I,J)=TOTX(I,J)+FX(I,J)
      TOTY(I,J)=TOTY(I,J)+FY(I,J)
  70  CONTINUE
C     TOTX=TOTX+FX
C     TOTY=TOTY+FY
      call exchange2d_mpi(fx,im,jm)
      call exchange2d_mpi(fy,im,jm)
      CALL VORT_CURL(FX,FY,DX,DY,DUM,DVM,IM,JM,CTEN)
      call exchange2d_mpi( cten,   im, jm )
C
      DO 80 I=2,IM
      DO 80 J=2,JM
      CTOT(I,J)=CTSURF(I,J)+CTBOT(I,J)+CPVF(I,J)+CJBAR(I,J)
     1   +CTEN(I,J)+CADV(I,J)
      TOTX(I,J)=TOTX(I,J)*DUM(I,J)*DUM(I-1,J)
      TOTY(I,J)=TOTY(I,J)*DVM(I,J)*DVM(I,J-1)
  80  CONTINUE
      call exchange2d_mpi( totx,   im, jm )
      call exchange2d_mpi( toty,   im, jm )
c
*/
	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			if (iid == 1){
				dmx = 0.5f*(d[j][i]+d[j][i-1]);
				dmy = 0.5f*(d[j][i]+d[j-1][i]);
			}else{
				dmx = 1.f;
				dmy = 1.f;
			}

			fx[j][i] = (uaf[j][i]*(h[j][i]+elf[j][i]
								  +h[j][i-1]+elf[j][i-1])
					    -uab[j][i]*(h[j][i]+elb[j][i]
								   +h[j][i-1]+elb[j][i-1]))
					   /(4.f*dte*dmx)*dum[j][i];

			fy[j][i] = (vaf[j][i]*(h[j][i]+elf[j][i]
								  +h[j-1][i]+elf[j-1][i])
						 -vab[j][i]*(h[j][i]+elb[j][i]
								    +h[j-1][i]+elb[j-1][i]))
					   /(4.f*dte*dmy)*dvm[j][i];


			totx[j][i] = totx[j][i]+fx[j][i];
			toty[j][i] = toty[j][i]+fy[j][i];
		}
	}

	exchange2d_mpi(fx, im, jm);
	exchange2d_mpi(fy, im, jm);
	vort_curl(fx, fy, dx, dy, dum, dvm, im, jm, cten);
	exchange2d_mpi(cten, im, jm);

	for (i = 1; i < im; i++){
		for (j = 1; j < jm; j++){
			ctot[j][i] = ctsurf[j][i]+ctbot[j][i]
						+cpvf[j][i]+cjbar[j][i]
						+cten[j][i]+cadv[j][i];
			totx[j][i] = totx[j][i]*dum[j][i]*dum[j][i-1];
			toty[j][i] = toty[j][i]*dvm[j][i]*dvm[j-1][i];

		}
	}
	exchange2d_mpi(totx, im, jm);
	exchange2d_mpi(toty, im, jm);

/*
      RETURN
      END
C
C
C --------------------------------------------------------------------
*/
	return;
}


void realvertvl(){

	int i,j,k;
    float dxr,dxl,dyt,dyb;
	float tps[j_size][i_size];
	float wr[k_size][j_size][i_size];

/*
 *dt: write again in advance.f dt = h + et
 *
 *et: assigned in advance.f (surface)
 *
 *w: assigned in advance.f (surface)
 *
 *u: read and write in advance.f
 *
 *v: read and write in advance.f
 *
 *etf: re-assigned in advance.f(mode_internal)
 *
 *etb: re-assigned in advance.f(mode_internal)
 */

/*
! calculate real w as wr
      do k=1,kb
        do j=1,jm
          do i=1,im
            wr(i,j,k)=0.
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				wr[k][j][i] = 0;	
			}
		}
	}

/*
      do k=1,kbm1
        do j=1,jm
          do i=1,im
            tps(i,j)=zz(k)*dt(i,j) + et(i,j)
          end do
        end do
        do j=2,jmm1
          do i=2,imm1
            dxr=2.0/(dx(i+1,j)+dx(i,j))
            dxl=2.0/(dx(i,j)+dx(i-1,j))
            dyt=2.0/(dy(i,j+1)+dy(i,j))
            dyb=2.0/(dy(i,j)+dy(i,j-1))
            wr(i,j,k)=0.5*(w(i,j,k)+w(i,j,k+1))+0.5*
     $                (u(i+1,j,k)*(tps(i+1,j)-tps(i,j))*dxr+
     $                 u(i,j,k)*(tps(i,j)-tps(i-1,j))*dxl+
     $                 v(i,j+1,k)*(tps(i,j+1)-tps(i,j))*dyt+
     $                 v(i,j,k)*(tps(i,j)-tps(i,j-1))*dyb)
     $                +(1.0+zz(k))*(etf(i,j)-etb(i,j))/dti2
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				tps[j][i] = zz[k]*dt[j][i] + et[j][i];	
			}
		}

		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				dxr = 2.f/(dx[j][i+1]+dx[j][i]);
				dxl = 2.f/(dx[j][i]+dx[j][i-1]);
				dyt = 2.f/(dy[j+1][i]+dy[j][i]);
				dyb = 2.f/(dy[j][i]+dy[j-1][i]);

				wr[k][j][i] = 0.5f*(w[k][j][i]+w[k+1][j][i])
							 +0.5f*(u[k][j][i+1]
									 *(tps[j][i+1]-tps[j][i])
									 *dxr
								   +u[k][j][i]
									 *(tps[j][i]-tps[j][i-1])
									 *dxl
								   +v[k][j+1][i]
									 *(tps[j+1][i]-tps[j][i])
									 *dyt
								   +v[k][j][i]
									 *(tps[j][i]-tps[j-1][i])
									 *dyb)
							 +(1.f+zz[k])*(etf[j][i]-etb[j][i])/dti2;
			}
		}
	}

//      call exchange3d_mpi(wr(:,:,1:kbm1),im,jm,kbm1)
	//exchange3d_mpi(wr, im, jm, kbm1);

/*
      do k=1,kb
        do i=1,im
          if(n_south.eq.-1) wr(i,1,k)=wr(i,2,k)
          if(n_north.eq.-1) wr(i,jm,k)=wr(i,jmm1,k)
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (i = 0; i < im; i++){
			if (n_south == -1)
				wr[k][0][i] = wr[k][1][i];
			if (n_north == -1)
				wr[k][jm-1][i] = wr[k][jmm1-1][i];
		}
	}


/*
      do k=1,kb
        do j=1,jm
          if(n_west.eq.-1) wr(1,j,k)=wr(2,j,k)
          if(n_east.eq.-1) wr(im,j,k)=wr(imm1,j,k)
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			if (n_west == -1)	
				wr[k][j][0] = wr[k][j][1];
			if (n_east == -1)
				wr[k][j][im-1] = wr[k][j][imm1-1];
		}
	}

/*
      do k=1,kbm1
        do j=1,jm
          do i=1,im
            wr(i,j,k)=fsm(i,j)*wr(i,j,k)
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				wr[k][j][i] = fsm[j][i]*wr[k][j][i];	
			}
		}
	}
}


/*
!_______________________________________________________________________
      subroutine vertvl
! calculates vertical velocity
      implicit none
      include 'pom.h'
      real xflux(im,jm,kb),yflux(im,jm,kb)
      integer i,j,k

      double precision vertvl_time_start_xsz
      double precision vertvl_time_end_xsz

      call call_time_start(vertvl_time_start_xsz)
*/

/* 
void vertvl_(float dt[][i_size], float u[][j_size][i_size],
		     float v[][j_size][i_size],float vfluxb[][i_size],
			 float vfluxf[][i_size], float w[][j_size][i_size],
			 float etf[][i_size],float etb[][i_size]){
*/
void vertvl(){

	//only modified -w 
	int i,j,k;
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
/*
 *dt: write again in advance.f dt = h + et
 *
 *u: read and write in advance.f
 *
 *v: read and write in advance.f
 *
 *vfluxb:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *vfluxf:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *w: assigned in advance.f (surface)
 *
 *etf: re-assigned in advance.f(mode_internal)
 *
 *etb: re-assigned in advance.f(mode_internal)
 */

/*	
! reestablish boundary conditions
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=.25e0*(dy(i,j)+dy(i-1,j))
     $                    *(dt(i,j)+dt(i-1,j))*u(i,j,k)
          end do
        end do
      end do
*/
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				xflux[k][j][i] = 0.25f*(dy[j][i]+dy[j][i-1])
								  *(dt[j][i]+dt[j][i-1])*u[k][j][i];
			}
		}
	}
/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            yflux(i,j,k)=.25e0*(dx(i,j)+dx(i,j-1))
     $                    *(dt(i,j)+dt(i,j-1))*v(i,j,k)
          end do
        end do
      end do
*/
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				yflux[k][j][i] = 0.25f*(dx[j][i]+dx[j-1][i])
								  *(dt[j][i]+dt[j-1][i])*v[k][j][i];
			}
		}
	}
/*
! note: if one wishes to include freshwater flux, the surface velocity
! should be set to vflux(i,j). See also change made to 2-D volume
! conservation equation which calculates elf
      do j=2,jmm1
        do i=2,imm1
          w(i,j,1)=0.5*(vfluxb(i,j)+vfluxf(i,j))
        end do
      end do
*/
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			w[0][j][i] = 0.5f*(vfluxb[j][i]+vfluxf[j][i]);
		}
	}
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            w(i,j,k+1)=w(i,j,k)
     $                +dz(k)*((xflux(i+1,j,k)-xflux(i,j,k)
     $                        +yflux(i,j+1,k)-yflux(i,j,k))
     $                        /(dx(i,j)*dy(i,j))
     $                        +(etf(i,j)-etb(i,j))/dti2)
          end do
        end do
      end do
*/
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				w[k+1][j][i] = w[k][j][i]
							  +dz[k]*((xflux[k][j][i+1]-xflux[k][j][i]
										+yflux[k][j+1][i]-yflux[k][j][i])
									   /(dx[j][i]*dy[j][i])
							          +(etf[j][i]-etb[j][i])/dti2);
			}
		}
	}
    return;
	
}


/*
      subroutine advq(qb,q,qf)
! calculates horizontal advection and diffusion, and vertical advection
! for turbulent quantities
      implicit none
      include 'pom.h'
      real qb(im_local,jm_local,kb),q(im_local,jm_local,kb)
      real qf(im_local,jm_local,kb)
      real xflux(im,jm,kb),yflux(im,jm,kb)
      integer i,j,k

      double precision advq_time_start_xsz
      double precision advq_time_end_xsz

      call call_time_start(advq_time_start_xsz)
*/

/*
void advq_(float qb[][j_size][i_size], float q[][j_size][i_size],
		  float qf[][j_size][i_size], float u[][j_size][i_size],
		  float dt[][i_size], float v[][j_size][i_size],
		  float aam[][j_size][i_size], float w[][j_size][i_size],
		  float etb[][i_size], float etf[][i_size]){
*/
void advq(float qb[][j_size][i_size], 
		  float q[][j_size][i_size],
		  float qf[][j_size][i_size]){

	//modify:
	//     -qf
	int i,j,k;
    float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
/*
 *u: read and write in advance.f
 *
 *v: read and write in advance.f
 *
 *dt: write again in advance.f dt = h + et
 *
 *aam: read and write in advance.f
 * 
 *w: assigned in advance.f (surface)
 *
 *etb: re-assigned in advance.f(mode_internal)
 *
 *etf: re-assigned in advance.f(mode_internal)
 */

//! do horizontal advection
	/*
      do k=2,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=.125e0*(q(i,j,k)+q(i-1,j,k))
     $                    *(dt(i,j)+dt(i-1,j))*(u(i,j,k)+u(i,j,k-1))
            yflux(i,j,k)=.125e0*(q(i,j,k)+q(i,j-1,k))
     $                    *(dt(i,j)+dt(i,j-1))*(v(i,j,k)+v(i,j,k-1))
          end do
        end do
      end do
	*/

	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i] = 0.125f*(q[k][j][i]+q[k][j][i-1])
									*(dt[j][i]+dt[j][i-1])
									*(u[k][j][i]+u[k-1][j][i]);
				yflux[k][j][i] = 0.125f*(q[k][j][i]+q[k][j-1][i])
									*(dt[j][i]+dt[j-1][i])
									*(v[k][j][i]+v[k-1][j][i]);
			}
		}
	}

//! do horizontal diffusion
/*
      do k=2,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=xflux(i,j,k)
     $                    -.25e0*(aam(i,j,k)+aam(i-1,j,k)
     $                            +aam(i,j,k-1)+aam(i-1,j,k-1))
     $                          *(h(i,j)+h(i-1,j))
     $                          *(qb(i,j,k)-qb(i-1,j,k))*dum(i,j)
     $                          /(dx(i,j)+dx(i-1,j))
            yflux(i,j,k)=yflux(i,j,k)
     $                    -.25e0*(aam(i,j,k)+aam(i,j-1,k)
     $                            +aam(i,j,k-1)+aam(i,j-1,k-1))
     $                          *(h(i,j)+h(i,j-1))
     $                          *(qb(i,j,k)-qb(i,j-1,k))*dvm(i,j)
     $                          /(dy(i,j)+dy(i,j-1))
            xflux(i,j,k)=.5e0*(dy(i,j)+dy(i-1,j))*xflux(i,j,k)
            yflux(i,j,k)=.5e0*(dx(i,j)+dx(i,j-1))*yflux(i,j,k)
          end do
        end do
      end do
*/
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i]=xflux[k][j][i]
								-0.25f*(aam[k][j][i]+aam[k][j][i-1]
										+aam[k-1][j][i]+aam[k-1][j][i-1])
									  *(h[j][i]+h[j][i-1])
									  *(qb[k][j][i]-qb[k][j][i-1])
									  *dum[j][i]
									  /(dx[j][i]+dx[j][i-1]);

				yflux[k][j][i]=yflux[k][j][i]
								-0.25f*(aam[k][j][i]+aam[k][j-1][i]
										+aam[k-1][j][i]+aam[k-1][j-1][i])
									  *(h[j][i]+h[j-1][i])
									  *(qb[k][j][i]-qb[k][j-1][i])
									  *dvm[j][i]/(dy[j][i]+dy[j-1][i]);

				xflux[k][j][i]=0.5f*(dy[j][i]+dy[j][i-1])
								   *xflux[k][j][i];

				yflux[k][j][i]=0.5f*(dx[j][i]+dx[j-1][i])
								   *yflux[k][j][i];
			}
		}
	}

    //call exchange3d_mpi(xflux(:,:,1:kbm1),im,jm,kbm1)
    //call exchange3d_mpi(yflux(:,:,1:kbm1),im,jm,kbm1)
    //exchange3d_mpi_xsz_(xflux,im,jm,0,kbm1-1);
    //exchange3d_mpi_bak(xflux,im,jm,0,kbm1-1);
    //exchange3d_mpi(xflux,im,jm,kbm1);
    //exchange3d_mpi_xsz_(yflux,im,jm,0,kbm1-1);
    //exchange3d_mpi_bak(yflux,im,jm,0,kbm1-1);
    //exchange3d_mpi(yflux,im,jm,kbm1);

//! do vertical advection, add flux terms, then step forward in time
/*
      do k=2,kbm1
        do j=2,jmm1
          do i=2,imm1
            qf(i,j,k)=(w(i,j,k-1)*q(i,j,k-1)-w(i,j,k+1)*q(i,j,k+1))
     $                 *art(i,j)/(dz(k)+dz(k-1))
     $                 +xflux(i+1,j,k)-xflux(i,j,k)
     $                 +yflux(i,j+1,k)-yflux(i,j,k)
            qf(i,j,k)=((h(i,j)+etb(i,j))*art(i,j)
     $                 *qb(i,j,k)-dti2*qf(i,j,k))
     $                /((h(i,j)+etf(i,j))*art(i,j))
          end do
        end do
      end do
*/

	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				qf[k][j][i]=(w[k-1][j][i]*q[k-1][j][i]
								-w[k+1][j][i]*q[k+1][j][i])
							  *art[j][i]
							  /(dz[k]+dz[k-1])
							+xflux[k][j][i+1]-xflux[k][j][i]
							+yflux[k][j+1][i]-yflux[k][j][i];

				qf[k][j][i]=((h[j][i]+etb[j][i])
								*art[j][i]*qb[k][j][i]
							  -dti2*qf[k][j][i])
							/((h[j][i]+etf[j][i])*art[j][i]);
			}
		}
	}

	return;
}


/*
	subroutine profq
! solve for q2 (twice the turbulent kinetic energy), q2l (q2 x turbulent
! length scale), km (vertical kinematic viscosity) and kh (vertical
! kinematic diffusivity), using a simplified version of the level 2 1/2
! model of Mellor and Yamada (1982)
! in this version, the Craig-Banner sub-model whereby breaking wave tke
! is injected into the surface is included. However, we use an
! analytical solution to the near surface tke equation to solve for q2
! at the surface giving the same result as C-B diffusion. The new scheme
! is simpler and more robust than the latter scheme
      implicit none
      include 'pom.h'
      real sm(im,jm,kb),sh(im,jm,kb),cc(im,jm,kb)
      real gh(im,jm,kb),boygr(im,jm,kb),dh(im,jm),stf(im,jm,kb)
      real prod(im,jm,kb)
      real a1,a2,b1,b2,c1
      real coef1,coef2,coef3,coef4,coef5
      real const1,e1,e2,ghc
      real p,sef,sp,tp
      real l0(im,jm)
      real cbcnst,surfl,shiw
      real utau2(im,jm)
      real df0,df1,df2
      integer i,j,k,ki

      data a1,b1,a2,b2,c1/0.92e0,16.6e0,0.74e0,10.1e0,0.08e0/
      data e1/1.8e0/,e2/1.33e0/
      data sef/1.e0/
      data cbcnst/100./surfl/2.e5/shiw/0.0/

*/

/*
void profq_(float etf[][i_size], float wusurf[][i_size],
			float wvsurf[][i_size],float wubot[][i_size],
			float wvbot[][i_size], float q2b[][j_size][i_size],
			float q2lb[][j_size][i_size], float u[][j_size][i_size],
			float v[][j_size][i_size], float km[][j_size][i_size], 
			float uf[][j_size][i_size], float vf[][j_size][i_size],
			float q2[][j_size][i_size], float dt[][i_size],
			float kh[][j_size][i_size], float t[][j_size][i_size],
			float s[][j_size][i_size], float rho[][j_size][i_size]){
*/
void profq(){

// modify:
//			uf, vf, 
//	+referencd: kq, km, kh, l(?)
//	+reference:	q2b, q2lb
	int i,j,k,ki;
    float a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    float ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
    float sm[k_size][j_size][i_size],sh[k_size][j_size][i_size];
    float cc[k_size][j_size][i_size];
    float gh[k_size][j_size][i_size],boygr[k_size][j_size][i_size];
	float dh[j_size][i_size],stf[k_size][j_size][i_size];
    float prod[k_size][j_size][i_size];
    float a1,a2,b1,b2,c1;
    float coef1,coef2,coef3,coef4,coef5;
    float const1,e1,e2,ghc;
    float p,sef,sp,tp;
    float l0[j_size][i_size];
    float cbcnst,surfl,shiw;
    float utau2[j_size][i_size];
    float df0,df1,df2;

	//it is only used in this function and each time it is first flush to a new num
	float dtef[k_size][j_size][i_size];
	
//we should change wubot! for it is changed in advave_ only! and referenced here and profu
//we should change wvbot! for it is changed in profv_ only!
//we should change km! for it is changed here and reference only in profv & profu
//we should change kh! for it is changed here and reference only in proft 

/*
 *etf: re-assigned in advance.f(mode_internal)
 *
 *wusurf:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *wvsurf:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *q2b: reference and modified in advance.f(mode_internal)
 *
 *q2: reference and modified in advance.f(mode_internal)
 *
 *q2lb: reference and modified in advance.f(mode_internal)
 *
 *u: read and write in advance.f
 *
 *v: read and write in advance.f
 *
 *uf: read in advance.f
 *
 *vf: read in advance.f
 *
 *dt: write again in advance.f dt = h + et
 *
 *t: reference and modified in advance.f(mode_internal)
 *
 *s: reference and modified in advance.f(mode_internal)
 *
 *rho: it is referenced and modified in baropg, and referenced here;
 *     it is also referenced in advance.f to call function dens, 
 *     so if we change all the contro logic, we can make it global
 */

/*
    data a1,b1,a2,b2,c1/0.92e0,16.6e0,0.74e0,10.1e0,0.08e0/
    data e1/1.8e0/,e2/1.33e0/
    data sef/1.e0/
    data cbcnst/100./surfl/2.e5/shiw/0.0/
*/
    a1=0.92f;b1=16.6f;a2=0.74f;b2=10.1f;c1=0.08f;
    e1=1.8e0f;e2=1.33e0f;
    sef=1.e0f;
    cbcnst=100.0f;surfl=2.0e5f;shiw=0.0f;

/*
      do j=1,jm
        do i=1,im
          dh(i,j)=h(i,j)+etf(i,j)
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = h[j][i]+etf[j][i];
		}
	}
/*
	do k=2,kbm1
        do j=1,jm
          do i=1,im
            a(i,j,k)=-dti2*(kq(i,j,k+1)+kq(i,j,k)+2.e0*umol)*.5e0
     $                /(dzz(k-1)*dz(k)*dh(i,j)*dh(i,j))
            c(i,j,k)=-dti2*(kq(i,j,k-1)+kq(i,j,k)+2.e0*umol)*.5e0
     $                /(dzz(k-1)*dz(k-1)*dh(i,j)*dh(i,j))
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k][j][i] = -dti2*(kq[k+1][j][i]
						           +kq[k][j][i]
								   +2.0f*umol)*0.5f
							 /(dzz[k-1]*dz[k]
							  *dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(kq[k-1][j][i]
						           +kq[k][j][i]
								   +2.0f*umol)*0.5f
						     /(dzz[k-1]*dz[k-1]
							  *dh[j][i]*dh[j][i]);

			}
		}
	}
	
/*	
! the following section solves the equation:
!     dti2*(kq*q2')' - q2*(2.*dti2*dtef+1.) = -q2b
*/
//! surface and bottom boundary conditions
      //const1=(16.6e0**(2.e0/3.e0))*sef;
      const1=powf(16.6e0f,(2.e0f/3.e0f))*sef;
	  

/*
! initialize fields that are not calculated on all boundaries
! but are later used there
*/
/*	
      do i=1,im
        ee(i,jm,1)=0.
        gg(i,jm,1)=0.
        l0(i,jm)=0.
      end do
*/
	for(i = 0; i < im; i++){
		ee[0][jm-1][i] = 0;
		gg[0][jm-1][i] = 0;
		l0[jm-1][i] = 0;
	}

/*
      do j=1,jm
        ee(im,j,1)=0.
        gg(im,j,1)=0.
        l0(im,j)=0.
      end do
*/
	for (j = 0; j < jm; j++){
		ee[0][j][im-1] = 0;	
		gg[0][j][im-1] = 0;
		l0[j][im-1] = 0;
	}
/*
      do i=1,im
        do j=1,jm
          do k=2,kbm1
            prod(i,j,k)=0.
          end do
        end do
      end do
*/
	for(i = 0; i < im; i++){
		for(j = 0; j < jm; j++){
			for(k = 1; k < kbm1; k++){
				prod[k][j][i] = 0.0f;
			}
		}
	}
/*	
      do j=1,jmm1
        do i=1,imm1
          utau2(i,j)=sqrt((.5e0*(wusurf(i,j)+wusurf(i+1,j)))**2
     $                   +(.5e0*(wvsurf(i,j)+wvsurf(i,j+1)))**2)
          ! wave breaking energy- a variant of Craig & Banner (1994)
          ! see Mellor and Blumberg, 2003.
          ee(i,j,1)=0.e0
          gg(i,j,1)=(15.8*cbcnst)**(2./3.)*utau2 
          ! surface length scale following Stacey (1999).
          l0(i,j)=surfl*utau2/grav
          uf(i,j,kb)=sqrt((.5e0*(wubot(i,j)+wubot(i+1,j)))**2
     $                   +(.5e0*(wvbot(i,j)+wvbot(i,j+1)))**2)*const1
        end do
      end do
*/
	for(j = 0; j < jmm1; j++){
		for(i = 0; i < imm1; i++){
			float tmpu_surf = 0.5f*(wusurf[j][i]+wusurf[j][i+1]);
			float tmpv_surf = 0.5f*(wvsurf[j][i]+wvsurf[j+1][i]);
			utau2[j][i] = sqrtf(tmpu_surf*tmpu_surf+tmpv_surf*tmpv_surf);

			ee[0][j][i] = 0.0;
			gg[0][j][i] = powf((15.8f*cbcnst),(2.0f/3.0f))*utau2[j][i];
			//comment by xsz: powf (float)powf
			l0[j][i] = surfl*utau2[j][i]/grav;

			float tmpu_bot = 0.5f*(wubot[j][i]+wubot[j][i+1]);
			float tmpv_bot = 0.5f*(wvbot[j][i]+wvbot[j+1][i]);

			uf[kb-1][j][i] = sqrtf((tmpu_bot*tmpu_bot)
								   +(tmpv_bot*tmpv_bot))
							 *const1;
		}
	}
/*
      call exchange2d_mpi(ee(:,:,1),im,jm)
      call exchange2d_mpi(gg(:,:,1),im,jm)
      call exchange2d_mpi(l0,im,jm)
*/
	exchange2d_mpi(ee[0], im, jm);
	exchange2d_mpi(gg[0], im, jm);
	exchange2d_mpi(l0, im, jm);
	
/*	
      do k=1,kbm1
        do j=1,jm
          do i=1,im
            tp=t(i,j,k)+tbias
            sp=s(i,j,k)+sbias
! calculate pressure in units of decibars
            p=grav*rhoref*(-zz(k)*h(i,j))*1.e-4
            cc(i,j,k)=1449.1e0+.00821e0*p+4.55e0*tp-.045e0*tp**2
     $                 +1.34e0*(sp-35.0e0)
            cc(i,j,k)=cc(i,j,k)
     $                 /sqrt((1.e0-.01642e0*p/cc(i,j,k))
     $                   *(1.e0-0.40e0*p/cc(i,j,k)**2))
          end do
        end do
      end do
*/
	for(k = 0; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				tp = t[k][j][i] + tbias;
				sp = s[k][j][i] + sbias;
				p = grav*rhoref*(-zz[k]*h[j][i])*1.0e-4f;
				cc[k][j][i] = 1449.1f+0.00821f*p+4.55f*tp-0.045f*(tp*tp)
								+1.34f*(sp-35.0f);
				//cc[k][j][i] = 1449.1f+0.00821f*p+4.55f*tp-0.045f*(powf(tp,2))
				//				+1.34f*(sp-35.0f);
				cc[k][j][i] = cc[k][j][i]
								/sqrtf((1.0f-0.01642f*p/cc[k][j][i])
									*(1.0f-0.4f*p/(cc[k][j][i]*cc[k][j][i])));
				//cc[k][j][i] = cc[k][j][i]
				//				/sqrtf((1.0f-0.01642f*p/cc[k][j][i])*
				//					(1.0f-0.4f*p/(powf(cc[k][j][i],2))));
				//comment by xsz: sqrtf
				
			}
		}
	}
	
/*
! calculate buoyancy gradient
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            q2b(i,j,k)=abs(q2b(i,j,k))
            q2lb(i,j,k)=abs(q2lb(i,j,k))
            boygr(i,j,k)=grav*(rho(i,j,k-1)-rho(i,j,k))
     $                    /(dzz(k-1)*h(i,j))
! *** note: comment out next line if dens does not include pressure
     $      +(grav**2)*2.e0/(cc(i,j,k-1)**2+cc(i,j,k)**2)
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				q2b[k][j][i] = ABS(q2b[k][j][i]);
				q2lb[k][j][i] = ABS(q2lb[k][j][i]);
				
				boygr[k][j][i] = grav*(rho[k-1][j][i]-rho[k][j][i])
									/(dzz[k-1]*h[j][i])
								+(grav*grav)*2.0f
									/((cc[k-1][j][i]*cc[k-1][j][i])
									 +(cc[k][j][i]*cc[k][j][i]));
			}
		}
	}
	
/*
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            l(i,j,k)=abs(q2lb(i,j,k)/(q2b(i,j,k)+small))
            if(z(k).gt.-0.5) l(i,j,k)=max(l(i,j,k),kappa*l0(i,j))
            gh(i,j,k)=(l(i,j,k)**2)*boygr(i,j,k)
     $           /(q2b(i,j,k)+small)
            gh(i,j,k)=min(gh(i,j,k),.028e0)
          end do
        end do
      end do

*/
	for(k = 1; k < kbm1; k++){
		for(j =0; j < jm; j++){
			for(i = 0; i < im; i++){
				l[k][j][i] = ABS(q2lb[k][j][i]/(q2b[k][j][i]+small));
				if (z[k] > -0.5f)
					l[k][j][i] = MAX(l[k][j][i],(kappa*l0[j][i]));
				gh[k][j][i] = (l[k][j][i]*l[k][j][i])
							  *boygr[k][j][i]/(q2b[k][j][i]+small);
				gh[k][j][i] = MIN(gh[k][j][i],0.028f);
			}
		}
	}
	
/*	
      do j=1,jm
        do i=1,im
          l(i,j,1)=kappa*l0(i,j)
          l(i,j,kb)=0.e0
          gh(i,j,1)=0.e0
          gh(i,j,kb)=0.e0
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			l[0][j][i] = kappa*l0[j][i];
			l[kb-1][j][i] = 0.0f;
			gh[0][j][i] = 0.0f;
			gh[kb-1][j][i] = 0.0f;
		}
	}
//! calculate production of turbulent kinetic energy:
/*
      do k=2,kbm1
        do j=2,jmm1
          do i=2,imm1
            prod(i,j,k)=km(i,j,k)*.25e0*sef
     $                   *((u(i,j,k)-u(i,j,k-1)
     $                      +u(i+1,j,k)-u(i+1,j,k-1))**2
     $                     +(v(i,j,k)-v(i,j,k-1)
     $                      +v(i,j+1,k)-v(i,j+1,k-1))**2)
     $                   /(dzz(k-1)*dh(i,j))**2
! add shear due to internal wave field
     $             -shiw*km(i,j,k)*boygr(i,j,k)
            prod(i,j,k)=prod(i,j,k)+kh(i,j,k)*boygr(i,j,k)
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				float tmpu = u[k][j][i]-u[k-1][j][i]+u[k][j][i+1]-u[k-1][j][i+1];
				float tmpv = v[k][j][i]-v[k-1][j][i]+v[k][j+1][i]-v[k-1][j+1][i];
				prod[k][j][i] = km[k][j][i]*0.25f*sef
									*((tmpu*tmpu)+(tmpv*tmpv))
									/((dzz[k-1]*dh[j][i])*(dzz[k-1]*dh[j][i]))
								-shiw*km[k][j][i]*boygr[k][j][i];
				prod[k][j][i] = prod[k][j][i]+kh[k][j][i]*boygr[k][j][i];
			}
		}
	}

//! note: Richardson # dep. dissipation correction (Mellor, 2001; Ezer,
//! 2000), depends on ghc the critical number (empirical -6 to -2) to
//! increase mixing
    ghc=-6.0f;
/*	  do k=1,kb
        do j=1,jm
          do i=1,im
            stf(i,j,k)=1.e0
! It is unclear yet if diss. corr. is needed when surf. waves are included.
!           if(gh(i,j,k).lt.0.e0)
!    $        stf(i,j,k)=1.0e0-0.9e0*(gh(i,j,k)/ghc)**1.5e0
!           if(gh(i,j,k).lt.ghc) stf(i,j,k)=0.1e0
            dtef(i,j,k)=sqrt(abs(q2b(i,j,k)))*stf(i,j,k)
     $                   /(b1*l(i,j,k)+small)
          end do
        end do
      end do
*/
	for(k = 0; k < kb; k++){
		for(j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				stf[k][j][i] = 1.0f;
/*
				if (gh[k][j][i] < 0.0){
					float tmp = gh[k][j][i]/ghc;
					//stf[k][j][i] = 1.0f-0.9f*(tmp)*sqrtf(tmp);
					stf[k][j][i] = 1.0f-0.9f*powf(tmp,1.5f);
					//comment by xsz: sqrtf
				}
				if (gh[k][j][i] < ghc)
					stf[k][j][i]=0.1f;
*/
				dtef[k][j][i] = sqrtf(ABS(q2b[k][j][i]))*stf[k][j][i]
								/(b1*l[k][j][i]+small);
			}
		}
	}

/*
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))
     $                      -(2.e0*dti2*dtef(i,j,k)+1.e0))
            ee(i,j,k)=a(i,j,k)*gg(i,j,k)
            gg(i,j,k)=(-2.e0*dti2*prod(i,j,k)+c(i,j,k)*gg(i,j,k-1)
     $                 -uf(i,j,k))*gg(i,j,k)
          end do
        end do
      end do
*/
/*
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0f/(a[k][j][i]+c[k][j][i]*(1.0f-ee[k-1][j][i])
					-(2.0f*dti2*dtef[k][j][i]+1.0f));
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (-2.0f*dti2*prod[k][j][i]+c[k][j][i]*gg[k-1][j][i]
						-uf[k][j][i])*gg[k][j][i];
			}
		}
	 }
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				float tmp = 1.0f/(a[k][j][i]
								 +c[k][j][i]
									*(1.0f-ee[k-1][j][i])
								 -(2.0f*dti2*dtef[k][j][i]+1.0f));
				ee[k][j][i] = a[k][j][i]*tmp;
				gg[k][j][i] = (-2.0f*dti2*prod[k][j][i]+c[k][j][i]*gg[k-1][j][i]
						-uf[k][j][i])*tmp;
			}
		}
		
	 }
	
/*	  
      do k=1,kbm1
        ki=kb-k
        do j=1,jm
          do i=1,im
            uf(i,j,ki)=ee(i,j,ki)*uf(i,j,ki+1)+gg(i,j,ki)
          end do
        end do
      end do
*/
/*
	for(k = 0; k < kbm1; k++){
		  ki = kb -k;
		  for(j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[ki-2][j][i] = ee[ki-2][j][i]*uf[ki+1-2][j][i]+gg[ki-2][j][i];
			}
		}  
	}
*/
	for(ki = kb-2; ki >= 0; ki--){
		for(j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[ki][j][i] = ee[ki][j][i]*uf[ki+1][j][i]+gg[ki][j][i];
			}
		}  
	}


//! the following section solves the equation:
//!     dti2(kq*q2l')' - q2l*(dti2*dtef+1.) = -q2lb
/*
	do j=1,jm
        do i=1,im
          vf(i,j,1)=0.
          vf(i,j,kb)=0.
          ee(i,j,2)=0.e0
          gg(i,j,2)=-kappa*z(2)*dh(i,j)*q2(i,j,2)
          vf(i,j,kb-1)=kappa*(1+z(kbm1))*dh(i,j)*q2(i,j,kbm1)
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			vf[0][j][i] = 0.0;
			vf[kb-1][j][i] =0.0;
			ee[1][j][i] = 0.0;
			gg[1][j][i] = -kappa*z[1]*dh[j][i]*q2[1][j][i];
			vf[kb-2][j][i] = kappa*(1.0f+z[kbm1-1])
				            *dh[j][i]*q2[kbm1-1][j][i];
		}
	}
	
/*
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            dtef(i,j,k)=dtef(i,j,k)
     $                   *(1.e0+e2*((1.e0/abs(z(k)-z(1))
     $                               +1.e0/abs(z(k)-z(kb)))
     $                                *l(i,j,k)/(dh(i,j)*kappa))**2)
          end do
        end do
      end do
*/


	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				float tmp = (1.0f/ABS(z[k]-z[0])+1.0f/ABS(z[k]-z[kb-1]))*l[k][j][i]/(dh[j][i]*kappa);
				//dtef[k][j][i] = dtef[k][j][i]
				//				*(1.0f+e2*tmp*tmp);
				//dtef[k][j][i] = dtef[k][j][i]*(1.0f+e2*powf(tmp,2.0f));
				tmp = 1.0f+e2*(tmp*tmp);
				dtef[k][j][i] = dtef[k][j][i]*tmp;
			}
		}
	}
	///////////////////////////////////
	//Ok!...//////////////////////////



/*	
      do k=3,kbm1
        do j=1,jm
          do i=1,im
            gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))
     $                      -(dti2*dtef(i,j,k)+1.e0))
            ee(i,j,k)=a(i,j,k)*gg(i,j,k)
            gg(i,j,k)=(dti2*(-prod(i,j,k)*l(i,j,k)*e1)
     $                 +c(i,j,k)*gg(i,j,k-1)-vf(i,j,k))*gg(i,j,k)
          end do
        end do
      end do
*/
	for(k = 2; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				float tmp = 1.0f/(a[k][j][i]
								 +c[k][j][i]*(1.0f-ee[k-1][j][i])
								 -(dti2*dtef[k][j][i]+1.0f));
				ee[k][j][i] = a[k][j][i]*tmp;
				gg[k][j][i] = (dti2*(-prod[k][j][i]
							         *l[k][j][i]*e1)
						       +c[k][j][i]*gg[k-1][j][i]
							   -vf[k][j][i])*tmp;
			}
		}
	}
/*	
      do k=1,kb-2
        ki=kb-k
        do j=1,jm
          do i=1,im
            vf(i,j,ki)=ee(i,j,ki)*vf(i,j,ki+1)+gg(i,j,ki)
          end do
        end do
      end do
*/
	/*
	for(k = 0; k < kb-2; k++){
		ki = kb-k;
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				vf[ki-2][j][i]=ee[ki-2][j][i]*vf[ki+1-2][j][i]+gg[ki-2][j][i];
			}
		}
	}
	*/

	for(ki = kb-2; ki > 0; ki--){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				vf[ki][j][i]=ee[ki][j][i]*vf[ki+1][j][i]+gg[ki][j][i];
			}
		}
	}

//! the following is to counter the problem of the ratio of two small
//! numbers (l = q2l/q2) or one number becoming negative. Two options are
//! included below. In this application, the second option, l was less
//! noisy when uf or vf is small
/*
	do k=2,kbm1
        do j=1,jm
          do i=1,im
!           if(uf(i,j,k).le.small.or.vf(i,j,k).le.small) then
!             uf(i,j,k)=small
!             vf(i,j,k)=0.1*dt(i,j)*small
!           end if
          uf(i,j,k)=abs(uf(i,j,k))
          vf(i,j,k)=abs(vf(i,j,k))
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
/*
				if (uf[k][j][i] <= small || vf[k][j][i] <= small){
					uf[k][j][i] = small;
					vf[k][j][i] = 0.1f*dt[j][i]*small;
				}
*/
				uf[k][j][i]=ABS(uf[k][j][i]);
				vf[k][j][i]=ABS(vf[k][j][i]);
				
			}
		}
	}
//! the following section solves for km and kh
/*
      coef4=18.e0*a1*a1+9.e0*a1*a2
      coef5=9.e0*a1*a2
*/
      coef4=18.0e0f*a1*a1+9.0e0f*a1*a2;
      coef5=9.0e0f*a1*a2;

//! note that sm and sh limit to infinity when gh approaches 0.0288
/*
	do k=1,kb
        do j=1,jm
          do i=1,im
            coef1=a2*(1.e0-6.e0*a1/b1*stf(i,j,k))
            coef2=3.e0*a2*b2/stf(i,j,k)+18.e0*a1*a2
            coef3=a1*(1.e0-3.e0*c1-6.e0*a1/b1*stf(i,j,k))
            sh(i,j,k)=coef1/(1.e0-coef2*gh(i,j,k))
            sm(i,j,k)=coef3+sh(i,j,k)*coef4*gh(i,j,k)
            sm(i,j,k)=sm(i,j,k)/(1.e0-coef5*gh(i,j,k))
          end do
        end do
      end do
*/

	for(k = 0; k < kb; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				coef1=a2*(1.0f-6.0f*a1/b1*stf[k][j][i]);
				coef2=3.0f*a2*b2/stf[k][j][i]+18.0f*a1*a2;
				coef3=a1*(1.0f-3.0f*c1-6.0f*a1/b1*stf[k][j][i]);

				sh[k][j][i]=coef1/(1.0f-coef2*gh[k][j][i]);
				sm[k][j][i]= coef3+sh[k][j][i]*coef4*gh[k][j][i];
				sm[k][j][i]=sm[k][j][i]/(1.0f-coef5*gh[k][j][i]); 
			}
		}
	}
	
//! there are 2 options for kq which, unlike km and kh, was not derived by
//! Mellor and Yamada but was purely empirical based on neutral boundary
//! layer data. The choice is whether or not it should be subject to the
//! stability factor, sh. Generally, there is not a great difference in
//! output
/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            prod(i,j,k)=l(i,j,k)*sqrt(abs(q2(i,j,k)))
            kq(i,j,k)=(prod(i,j,k)*.41e0*sh(i,j,k)+kq(i,j,k))*.5e0
!            kq(i,j,k)=(prod(i,j,k)*.20+kq(i,j,k))*.5e0
            km(i,j,k)=(prod(i,j,k)*sm(i,j,k)+km(i,j,k))*.5e0
            kh(i,j,k)=(prod(i,j,k)*sh(i,j,k)+kh(i,j,k))*.5e0
          end do
        end do
      end do
*/
	for(k =0; k < kb; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				prod[k][j][i]=l[k][j][i]*sqrtf(ABS(q2[k][j][i]));
				kq[k][j][i]=(prod[k][j][i]*0.41f*sh[k][j][i]
							+kq[k][j][i])*0.5f;
				km[k][j][i]=(prod[k][j][i]*sm[k][j][i]
							+km[k][j][i])*0.5f;
				kh[k][j][i]=(prod[k][j][i]*sh[k][j][i]
							+kh[k][j][i])*0.5f;
			}
		}
	}

      //call exchange3d_mpi(km,im,jm,kb)
    exchange3d_mpi(km,im,jm,kb);
      //call exchange3d_mpi(kh,im,jm,kb)
    exchange3d_mpi(kh,im,jm,kb);

//! cosmetics: make boundr. values as interior (even if not used, printout
//! may show strange values)
/*    
      do k=1,kb
        do i=1,im
          if(n_north.eq.-1) then
            km(i,jm,k)=km(i,jmm1,k)*fsm(i,jm)
            kh(i,jm,k)=kh(i,jmm1,k)*fsm(i,jm)
          end if
          if(n_south.eq.-1) then
            km(i,1,k)=km(i,2,k)*fsm(i,1)
            kh(i,1,k)=kh(i,2,k)*fsm(i,1)
          end if
        end do
        do j=1,jm
          if(n_east.eq.-1) then
            km(im,j,k)=km(imm1,j,k)*fsm(im,j)
            kh(im,j,k)=kh(imm1,j,k)*fsm(im,j)
          end if
          if(n_west.eq.-1) then
            km(1,j,k)=km(2,j,k)*fsm(1,j)
            kh(1,j,k)=kh(2,j,k)*fsm(1,j)
          end if
        end do
      end do
*/
	for(k = 0; k < kb; k++){
		for(i = 0; i < im; i++){
			if(n_north == -1){
				km[k][jm-1][i] = km[k][jmm1-1][i]*fsm[jm-1][i];
				kh[k][jm-1][i] = kh[k][jmm1-1][i]*fsm[jm-1][i];
			}
			if(n_south == -1){
				km[k][0][i] = km[k][1][i]*fsm[0][i];
				kh[k][0][i] = kh[k][1][i]*fsm[0][i];
			}
		}
	
		for(j = 0; j < jm; j++){
			if(n_east == -1){
				km[k][j][im-1] = km[k][j][imm1-1]*fsm[j][im-1];
				kh[k][j][im-1] = kh[k][j][imm1-1]*fsm[j][im-1];
			}
			if(n_west == -1){
				km[k][j][0] = km[k][j][1]*fsm[j][0];
				kh[k][j][0] = kh[k][j][1]*fsm[j][0];
			}
		}
	}

	return;
}


/*
      subroutine advt1(fb,f,fclim,ff,xflux,yflux, var)
! integrate conservative scalar equations
! this is centred scheme, as originally provide in POM (previously
! called advt)
      implicit none
      include 'pom.h'
      real fb(im,jm,kb),f(im,jm,kb),fclim(im,jm,kb),ff(im,jm,kb)
      real xflux(im,jm,kb),yflux(im,jm,kb)
      integer i,j,k
      real relax !lyo:relax

      character(len=1), intent(in) :: var

*/

/*
void advt1_(float fb[][j_size][i_size], float f[][j_size][i_size],
		   float fclim[][j_size][i_size], float ff[][j_size][i_size],
		   float dt[][i_size], float u[][j_size][i_size],
		   float v[][j_size][i_size], float aam[][j_size][i_size],
		   float w[][j_size][i_size], float etb[][i_size],
		   float etf[][i_size]){
*/
void advt1(float fb[][j_size][i_size], 
		   float f[][j_size][i_size],
		   float fclim[][j_size][i_size], 
		   float ff[][j_size][i_size],
		   char var){

// modify :
//     +reference f fb /only solve the boundary 
//     -          ff
	int i,j,k;
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	float zflux[k_size][j_size][i_size];
	float relax;
/*
 *dt: write again in advance.f dt = h + et
 *
 *u: read and write in advance.f
 *
 *v: read and write in advance.f
 *
 *aam: read and write in advance.f
 *
 *w: assigned in advance.f (surface)
 *
 *zflux: not used in other file, I believe it is a temp array
 *
 *etb: re-assigned in advance.f(mode_internal)
 *
 *etf: re-assigned in advance.f(mode_internal)
 */

/*
      do j=1,jm
        do i=1,im
           f(i,j,kb)=f(i,j,kbm1)
           fb(i,j,kb)=fb(i,j,kbm1)
        end do
      end do
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f[kb-1][j][i] = f[kbm1-1][j][i];
			fb[kb-1][j][i] = fb[kbm1-1][j][i];
		}
	}

//! do advective fluxes
/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=.25e0*((dt(i,j)+dt(i-1,j))
     $                          *(f(i,j,k)+f(i-1,j,k))*u(i,j,k))
            yflux(i,j,k)=.25e0*((dt(i,j)+dt(i,j-1))
     $                          *(f(i,j,k)+f(i,j-1,k))*v(i,j,k))
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i]=0.25f*((dt[j][i]+dt[j][i-1])*(f[k][j][i]+f[k][j][i-1])*u[k][j][i]);	
				yflux[k][j][i]=0.25f*((dt[j][i]+dt[j-1][i])*(f[k][j][i]+f[k][j-1][i])*v[k][j][i]);

			}
		}
	}

//! add diffusive fluxes
/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            fb(i,j,k)=fb(i,j,k)-fclim(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] = fb[k][j][i]-fclim[k][j][i];	
			}
		}
	}

/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=xflux(i,j,k)
     $                    -.5e0*(aam(i,j,k)+aam(i-1,j,k))
     $                         *(h(i,j)+h(i-1,j))*tprni
     $                         *(fb(i,j,k)-fb(i-1,j,k))*dum(i,j)
     $                         /(dx(i,j)+dx(i-1,j))
            yflux(i,j,k)=yflux(i,j,k)
     $                    -.5e0*(aam(i,j,k)+aam(i,j-1,k))
     $                         *(h(i,j)+h(i,j-1))*tprni
     $                         *(fb(i,j,k)-fb(i,j-1,k))*dvm(i,j)
     $                         /(dy(i,j)+dy(i,j-1))
            xflux(i,j,k)=.5e0*(dy(i,j)+dy(i-1,j))*xflux(i,j,k)
            yflux(i,j,k)=.5e0*(dx(i,j)+dx(i,j-1))*yflux(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i]=xflux[k][j][i]-0.5f*(aam[k][j][i]+aam[k][j][i-1])*(h[j][i]+h[j][i-1])*tprni*(fb[k][j][i]-fb[k][j][i-1])*dum[j][i]/(dx[j][i]+dx[j][i-1]);
				yflux[k][j][i]=yflux[k][j][i]-0.5f*(aam[k][j][i]+aam[k][j-1][i])*(h[j][i]+h[j-1][i])*tprni*(fb[k][j][i]-fb[k][j-1][i])*dvm[j][i]/(dy[j][i]+dy[j-1][i]);
				xflux[k][j][i]=0.5f*(dy[j][i]+dy[j][i-1])*xflux[k][j][i];
				yflux[k][j][i]=0.5f*(dx[j][i]+dx[j-1][i])*yflux[k][j][i];

			}
		}
	}


/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            fb(i,j,k)=fb(i,j,k)+fclim(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] += fclim[k][j][i];	
			}
		}
	}

//! do vertical advection
/*
 *   do j=2,jmm1
 *     do i=2,imm1
 *       if ( var == 'T' ) zflux(i,j,1)=tsurf(i,j)*w(i,j,1)*art(i,j)
 *		 if ( var == 'S' ) zflux(i,j,1)=0.e0 
 *		 zflux(i,j,kb)=0.e0
 *	   end do
 *   end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			if (var == 'T'){
				zflux[0][j][i] = tsurf[j][i]*w[0][j][i]*art[j][i];	
			}
			else if (var == 'S')
				zflux[0][j][i] = 0;

			zflux[kb-1][j][i]=0;
		}
	}

/*
      do k=2,kbm1
        do j=2,jmm1
          do i=2,imm1
            zflux(i,j,k)=.5e0*(f(i,j,k-1)+f(i,j,k))*w(i,j,k)*art(i,j)
          end do
        end do
      end do
*/
	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				zflux[k][j][i]=0.5f*(f[k-1][j][i]+f[k][j][i])*w[k][j][i]*art[j][i];	
			}
		}
	}

//! add net horizontal fluxes and then step forward in time
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            relax=1.586e-8*(1.e0-exp(zz(k)*h(i,j)*5.e-4)) !730 days, 2000m
!           relax=3.171e-8*(1.e0-exp(zz(k)*h(i,j)*1.e-3)) !365 days, 1000m
!           relax=6.430e-8*(1.e0-exp(zz(k)*h(i,j)*2.e-3)) !180 days,  500m
!           relax=0.0                                     !no relaxation
            ff(i,j,k)=xflux(i+1,j,k)-xflux(i,j,k)
     $                 +yflux(i,j+1,k)-yflux(i,j,k)
     $                 +(zflux(i,j,k)-zflux(i,j,k+1))/dz(k)
     $                 -relax*fclim(i,j,k)*dt(i,j)*art(i,j)
            ff(i,j,k)=(fb(i,j,k)*(h(i,j)+etb(i,j))*art(i,j)
     $                 -dti2*ff(i,j,k))
     $                 /((h(i,j)+etf(i,j))*art(i,j)*(1.+relax*dti2))
          end do
        end do
      end do
*/

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				//!730 days, 2000m
				relax = 1.586e-8f*(1.e0f-expf(zz[k]*h[j][i]*5.e-4f));
				//!365 days, 1000m
                //!relax=3.171e-8*(1.e0-exp(zz(k)*h(i,j)*1.e-3)) 
				//!180 days,  500m
                //!relax=6.430e-8*(1.e0-exp(zz(k)*h(i,j)*2.e-3)) 
				//!no relaxation
                //!relax=0.0                                     
				ff[k][j][i] = xflux[k][j][i+1]-xflux[k][j][i]
							 +yflux[k][j+1][i]-yflux[k][j][i]
							 +(zflux[k][j][i]-zflux[k+1][j][i])/dz[k]
							 -relax*fclim[k][j][i]*dt[j][i]*art[j][i];

				/*
				ff[k][j][i] = (fb[k][j][i]
								*(h[j][i]+etb[j][i])*art[j][i]
							   -dti2*ff[k][j][i])
							  /((h[j][i]+etf[j][i])*art[j][i]
								*(1.f+relax*dti2));
				*/
				array_3d_tmp1[k][j][i] = fb[k][j][i];
								//*(h[j][i]+etb[j][i]);//*art[j][i];

				ff[k][j][i] = (fb[k][j][i]
								*(h[j][i]+etb[j][i])*art[j][i]
							   -dti2*ff[k][j][i])
							  /((h[j][i]+etf[j][i])*art[j][i]
								*(1.f+relax*dti2));

				if (var == 'T'){
					if (array_3d_tmp1[k][j][i] != array_3d_tmp3[k][j][i]){
						printf("ff[%d][%d][%d] = %35.25f, tmp3 = %35.25f\n",
								k, j, i, array_3d_tmp1[k][j][i], array_3d_tmp3[k][j][i]);	
					}
				}


			}
		}
	}
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				//if (var == 'T'){
				//	if (ff[k][j][i] != array_3d_tmp3[k][j][i]){
				//		printf("ff[%d][%d][%d] = %35.25f, tmp3 = %35.25f\n",
				//				k, j, i, ff[k][j][i], array_3d_tmp3[k][j][i]);	
				//	}
				//}
			}
		}
	}

	return;
}

/*
void advt2_(float fb[][j_size][i_size],float f[][j_size][i_size],
		   float fclim[][j_size][i_size],float ff[][j_size][i_size],
		   float etb[][i_size], float u[][j_size][i_size],
		   float v[][j_size][i_size], float etf[][i_size], 
		   float aam[][j_size][i_size], float w[][j_size][i_size],
		   float dt[][i_size]){
*/
void advt2(float fb[][j_size][i_size],float f[][j_size][i_size],
		   float fclim[][j_size][i_size],float ff[][j_size][i_size],
		   char var){

/*
! <-- write fb(only boundary), ff

! integrate conservative scalar equations
! this is a first-order upstream scheme, which reduces implicit
! diffusion using the Smolarkiewicz iterative upstream scheme with an
! antidiffusive velocity
! it is based on the subroutines of Gianmaria Sannino (Inter-universityi
! Computing Consortium, Rome, Italy) and Vincenzo Artale (Italian
! National Agency for New Technology and Environment, Rome, Italy)
      implicit none
      include 'pom.h'
      real fb(im_local,jm_local,kb),f(im_local,jm_local,kb)
      real fclim(im_local,jm_local,kb),ff(im_local,jm_local,kb)
      real xflux(im,jm,kb),yflux(im,jm,kb)
      real fbmem(im,jm,kb),eta(im,jm)
      real xmassflux(im,jm,kb),ymassflux(im,jm,kb),zwflux(im,jm,kb)
      integer i,j,k,itera
      double precision smol_adif_time_start_xsz
      double precision smol_adif_time_end_xsz


      double precision advt2_time_start_xsz
      double precision advt2_time_end_xsz

      call call_time_start(advt2_time_start_xsz)
! calculate horizontal mass fluxes
*/
	int i, j, k, itera;	
	float xflux[k_size][j_size][i_size];
	float yflux[k_size][j_size][i_size];
	float zflux[k_size][j_size][i_size];

	float fbmem[k_size][j_size][i_size];
	float eta[j_size][i_size];

	float xmassflux[k_size][j_size][i_size];
	float ymassflux[k_size][j_size][i_size];
	float zwflux[k_size][j_size][i_size];

	/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            xmassflux(i,j,k)=0.e0
            ymassflux(i,j,k)=0.e0
          end do
        end do
      end do
	*/

	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				xmassflux[k][j][i] = 0.0;
				ymassflux[k][j][i] = 0.0;
			}
		}
	}
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,im
            xmassflux(i,j,k)=0.25e0*(dy(i-1,j)+dy(i,j))
     $                             *(dt(i-1,j)+dt(i,j))*u(i,j,k)
          end do
        end do

        do j=2,jm
          do i=2,imm1
            ymassflux(i,j,k)=0.25e0*(dx(i,j-1)+dx(i,j))
     $                             *(dt(i,j-1)+dt(i,j))*v(i,j,k)
          end do
        end do
      end do

*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < im; i++){
				xmassflux[k][j][i] = 0.25f*(dy[j][i-1]+dy[j][i])*(dt[j][i-1]+dt[j][i])*u[k][j][i];
			}
		}
		for (j = 1; j < jm; j++){
			for (i = 1; i < imm1; i++){
				ymassflux[k][j][i] = 0.25f*(dx[j-1][i]+dx[j][i])*(dt[j-1][i]+dt[j][i])*v[k][j][i];
			}
		}
	}
	
/*
      do j=1,jm
        do i=1,im
          fb(i,j,kb)=fb(i,j,kbm1)
          eta(i,j)=etb(i,j)
        end do
      end do
*/
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			fb[kb-1][j][i] = fb[kbm1-1][j][i];
			eta[j][i] = etb[j][i];
		}	
	}
/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            zwflux(i,j,k)=w(i,j,k)
            fbmem(i,j,k)=fb(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				zwflux[k][j][i] = w[k][j][i];	
				fbmem[k][j][i] = fb[k][j][i];
			}
		}
	}
/*
//! start Smolarkiewicz scheme
      do itera=1,nitera

//! upwind advection scheme
        do k=1,kbm1
          do j=2,jm
            do i=2,im
              xflux(i,j,k)=0.5e0
     $                      *((xmassflux(i,j,k)+abs(xmassflux(i,j,k)))
     $                        *fbmem(i-1,j,k)+
     $                        (xmassflux(i,j,k)-abs(xmassflux(i,j,k)))
     $                        *fbmem(i,j,k))

              yflux(i,j,k)=0.5e0
     $                      *((ymassflux(i,j,k)+abs(ymassflux(i,j,k)))
     $                        *fbmem(i,j-1,k)+
     $                        (ymassflux(i,j,k)-abs(ymassflux(i,j,k)))
     $                        *fbmem(i,j,k))
            end do
          end do
        end do

        do j=2,jmm1
          do i=2,imm1
            zflux(i,j,1)=0.e0
!            if(itera.eq.1) zflux(i,j,1)=w(i,j,1)*f(i,j,1)*art(i,j)
!     for rivers 2010/5/08 ayumi
            if (itera == 1 ) then
               if ( var == 'T' ) 
     $              zflux(i,j,1)=tsurf(i,j)*w(i,j,1)*art(i,j)
               if ( var == 'S' ) 
     $              zflux(i,j,1)=0.e0
            endif
            zflux(i,j,kb)=0.e0
          end do
        end do


        do k=2,kbm1
          do j=2,jmm1
            do i=2,imm1
              zflux(i,j,k)=0.5e0
     $                      *((zwflux(i,j,k)+abs(zwflux(i,j,k)))
     $                       *fbmem(i,j,k)+
     $                        (zwflux(i,j,k)-abs(zwflux(i,j,k)))
     $                       *fbmem(i,j,k-1))
              zflux(i,j,k)=zflux(i,j,k)*art(i,j)
            end do
          end do
        end do

! add net advective fluxes and step forward in time
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
              ff(i,j,k)=xflux(i+1,j,k)-xflux(i,j,k)
     $                 +yflux(i,j+1,k)-yflux(i,j,k)
     $                 +(zflux(i,j,k)-zflux(i,j,k+1))/dz(k)
              ff(i,j,k)=(fbmem(i,j,k)*(h(i,j)+eta(i,j))*art(i,j)
     $                   -dti2*ff(i,j,k))/((h(i,j)+etf(i,j))*art(i,j))
            end do
          end do
        end do

        ! next line added on 22-Jul-2009 by Raffaele Bernardello
        call exchange3d_mpi(ff(:,:,1:kbm1),im,jm,kbm1)

! calculate antidiffusion velocity
        call smol_adif(xmassflux,ymassflux,zwflux,ff)

        do j=1,jm
          do i=1,im
            eta(i,j)=etf(i,j)
            do k=1,kb
              fbmem(i,j,k)=ff(i,j,k)
            end do
          end do
        end do

! end of Smolarkiewicz scheme
      end do
*/
	for (itera = 1; itera <= nitera; itera++){
		for(k = 0; k < kbm1; k++){
			for (j = 1; j < jm; j++){
				for (i = 1; i < im; i++){
					xflux[k][j][i] = 0.5f*((xmassflux[k][j][i] + (ABS(xmassflux[k][j][i])))*fbmem[k][j][i-1] + (xmassflux[k][j][i]-(ABS(xmassflux[k][j][i])))*fbmem[k][j][i]);
					yflux[k][j][i] = 0.5f*((ymassflux[k][j][i] + (ABS(ymassflux[k][j][i])))*fbmem[k][j-1][i] + (ymassflux[k][j][i]-(ABS(ymassflux[k][j][i])))*fbmem[k][j][i]);

				}
			}
		}
		

		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				zflux[0][j][i] = 0;
				if (itera == 1){
					if (var == 'T')
						zflux[0][j][i] = tsurf[j][i]*w[0][j][i]*art[j][i];
					else if (var == 'S')
						zflux[0][j][i] = 0;
				}
				zflux[kb-1][j][i] = 0;
			}
		}

		for (k = 1; k < kbm1; k++){
			for (j = 1; j < jmm1; j++){
				for (i = 1; i < imm1; i++){
					zflux[k][j][i] = 0.5f*((zwflux[k][j][i] + (ABS(zwflux[k][j][i])))*fbmem[k][j][i] + (zwflux[k][j][i] - (ABS(zwflux[k][j][i])))*fbmem[k-1][j][i]);
					zflux[k][j][i] *= art[j][i]; 
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 1; j < jmm1; j++){
				for (i = 1; i < imm1; i++){
					ff[k][j][i] = xflux[k][j][i+1] - xflux[k][j][i]
								 +yflux[k][j+1][i] - yflux[k][j][i] 
								 +(zflux[k][j][i]-zflux[k+1][j][i])/dz[k];
					ff[k][j][i] = (fbmem[k][j][i]
									*(h[j][i]+eta[j][i])*art[j][i] 
								   -dti2*ff[k][j][i])
								  /((h[j][i]+etf[j][i])*art[j][i]);
						
				}
			}
		}

/*
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				for (k = 0; k < kbm1; k++){
					//ff[k][j][i] = xflux[k][j][i+1] - xflux[k][j][i] + yflux[k][j+1][i] - yflux[k][j][i] + (zflux[k][j][i]-zflux[k+1][j][i])/dz[k];
					ff[k][j][i] = (fbmem[k][j][i]*((double)((h[j][i]+eta[j][i])*art[j][i])) - dti2*ff[k][j][i])/((double)((h[j][i]+etf[j][i])*art[j][i]));

						
				}
			}
		}
*/
		//exchange3d_mpi_xsz_((void*)ff, im, jm, 0, kbm1-1);
		//exchange3d_mpi_bak((void*)ff, im, jm, 0, kbm1-1);
		exchange3d_mpi(ff, im, jm, kbm1);

		/*
		smol_adif_((void*)xmassflux, (void*)ymassflux, (void*)zwflux,
					(void*)ff, (void*)dt);
		*/
		smol_adif(xmassflux, ymassflux, zwflux, ff);
					
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				eta[j][i] = etf[j][i];	
				for (k = 0; k < kb; k++){
					fbmem[k][j][i] = ff[k][j][i];	
				}
			}
		}

	}

//! add horizontal diffusive fluxes
/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            fb(i,j,k)=fb(i,j,k)-fclim(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] = fb[k][j][i] - fclim[k][j][i];	
			}
		}
	}
/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            xmassflux(i,j,k)=0.5e0*(aam(i,j,k)+aam(i-1,j,k))
            ymassflux(i,j,k)=0.5e0*(aam(i,j,k)+aam(i,j-1,k))
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xmassflux[k][j][i] = 0.5f*(aam[k][j][i] + aam[k][j][i-1]);	
				ymassflux[k][j][i] = 0.5f*(aam[k][j][i] + aam[k][j-1][i]);	
			}
		}	
	}
/*
      do k=1,kbm1
        do j=2,jm
          do i=2,im
           xflux(i,j,k)=-xmassflux(i,j,k)*(h(i,j)+h(i-1,j))*tprni
     $                   *(fb(i,j,k)-fb(i-1,j,k))*dum(i,j)
     $                   *(dy(i,j)+dy(i-1,j))*0.5e0/(dx(i,j)+dx(i-1,j))
           yflux(i,j,k)=-ymassflux(i,j,k)*(h(i,j)+h(i,j-1))*tprni
     $                   *(fb(i,j,k)-fb(i,j-1,k))*dvm(i,j)
     $                   *(dx(i,j)+dx(i,j-1))*0.5e0/(dy(i,j)+dy(i,j-1))
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
			for (i = 1; i < im; i++){
				xflux[k][j][i] = -xmassflux[k][j][i]*(h[j][i]+h[j][i-1])*tprni*(fb[k][j][i]-fb[k][j][i-1])*dum[j][i]*(dy[j][i]+dy[j][i-1])*0.5f/(dx[j][i]+dx[j][i-1]);
				yflux[k][j][i] = -ymassflux[k][j][i]*(h[j][i]+h[j-1][i])*tprni*(fb[k][j][i]-fb[k][j-1][i])*dvm[j][i]*(dx[j][i]+dx[j-1][i])*0.5f/(dy[j][i]+dy[j-1][i]);
			}
		}
	}

/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            fb(i,j,k)=fb(i,j,k)+fclim(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				fb[k][j][i] += fclim[k][j][i];	
			}
		}
	}
	

//! add net horizontal fluxes and step forward in time
/*
      do j=2,jmm1
        do i=2,imm1
          do k=1,kbm1
            ff(i,j,k)=ff(i,j,k)-dti2*(xflux(i+1,j,k)-xflux(i,j,k)
     $                               +yflux(i,j+1,k)-yflux(i,j,k))
     $                           /(h(i,j)+etf(i,j))*art(i,j))
          end do
        end do
      end do
*/
	for (j = 1; j < jmm1; j++){
		for (i = 1; i < imm1; i++){
			for (k = 0; k < kbm1; k++){
				ff[k][j][i] = ff[k][j][i]
							 -dti2*(xflux[k][j][i+1]
								   -xflux[k][j][i]
								   +yflux[k][j+1][i]
								   -yflux[k][j][i])
								/((h[j][i]+etf[j][i])*art[j][i]);	
			}
		}
	}
	
	

    return;
}



void smol_adif(float xmassflux[][j_size][i_size], 
			   float ymassflux[][j_size][i_size], 
			   float zwflux[][j_size][i_size],
			   float ff[][j_size][i_size]){


/*
! calculate the antidiffusive velocity used to reduce the numerical
! diffusion associated with the upstream differencing scheme
! this is based on a subroutine of Gianmaria Sannino (Inter-university
! Computing Consortium, Rome, Italy) and Vincenzo Artale (Italian
! National Agency for New Technology and Environment, Rome, Italy)
*/
/*
    float ff(im_local,jm_local,kb)
    float xmassflux(im,jm,kb),ymassflux(im,jm,kb),zwflux(im,jm,kb)
    float mol,abs_1,abs_2
    float value_min,epsilon
    float udx,u2dt,vdy,v2dt,wdz,w2dt
	float dz(kb)
*/
    int i,j,k;
    float mol,abs_1,abs_2;
    float udx,u2dt,vdy,v2dt,wdz,w2dt;
    const float value_min=1.e-9,epsilon=1.0e-14;

	//int kb = *kb_in;
	//int jm = *jm_in;
	//int im = *im_in;
	//int kbm1 = *kbm1_in;
	//int imm1 = *imm1_in;
	//int jmm1 = *jmm1_in;

	//float sw = *sw_in;
	//float dti2 = *dti2_in;



// apply temperature and salinity mask
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
#pragma simd
#pragma vector
			for (i = 0; i < im ;i++){
				ff[k][j][i] *= fsm[j][i];
			}
		}
	}

//! recalculate mass fluxes with antidiffusion velocity
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
#pragma simd
#pragma vector
			for (i = 1; i < im; i++){
				if (ff[k][j][i] < value_min || ff[k][j][i-1] < value_min)
					xmassflux[k][j][i] = 0;
				else{
					udx = ABS(xmassflux[k][j][i]);
					u2dt = dti2*xmassflux[k][j][i]
							   *xmassflux[k][j][i]
							   *2.0f/(aru[j][i]
									   *(dt[j][i-1]+dt[j][i]));

					mol = (ff[k][j][i]-ff[k][j][i-1])
						 /(ff[k][j][i-1]+ff[k][j][i]+epsilon);

					xmassflux[k][j][i] = (udx-u2dt)*mol*sw;

					if (ABS(udx) < ABS(u2dt))
						xmassflux[k][j][i] = 0;
				}
			}
		}
	}

	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jm; j++){
#pragma simd
#pragma vector
			for (i = 1; i < imm1; i++){
				if (ff[k][j][i] < value_min 
				  ||ff[k][j-1][i] < value_min)
					ymassflux[k][j][i] = 0;
				else{
					vdy = ABS(ymassflux[k][j][i]);
					v2dt = dti2*ymassflux[k][j][i]
							   *ymassflux[k][j][i]
							   *2.0f/(arv[j][i]
									   *(dt[j-1][i]+dt[j][i]));

					mol = (ff[k][j][i]-ff[k][j-1][i])
						 /(ff[k][j-1][i]+ff[k][j][i]+epsilon);

					ymassflux[k][j][i] = (vdy-v2dt)*mol*sw;

					if (ABS(vdy) < ABS(v2dt))
						ymassflux[k][j][i] = 0;
				}
			}
		}
	}

	for (k = 1; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
#pragma simd
#pragma vector
			for (i = 1; i < imm1; i++){
				if (ff[k][j][i] < value_min 
				  ||ff[k-1][j][i] < value_min)

					zwflux[k][j][i] = 0;

				else{
					wdz = ABS(zwflux[k][j][i]);	
					w2dt = dti2*zwflux[k][j][i]
							   *zwflux[k][j][i]
							   /(dzz[k-1]*dt[j][i]);

					mol = (ff[k-1][j][i]-ff[k][j][i])
						 /(ff[k][j][i]+ff[k-1][j][i]+epsilon);

					zwflux[k][j][i] = (wdz-w2dt)*mol*sw;

					if (ABS(wdz) < ABS(w2dt))
						zwflux[k][j][i] = 0;
				}
			}
		}
	}

    return;
}


/*
!_______________________________________________________________________
      subroutine proft(f,wfsurf,fsurf,nbc)
! solves for vertical diffusion of temperature and salinity using method
! described by Richmeyer and Morton (1967)
! note: wfsurf and swrad are negative values when water column is
! warming or salt is being added
      implicit none
      include 'pom.h'
      real f(im_local,jm_local,kb)
      real wfsurf(im_local,jm_local),fsurf(im_local,jm_local)
      integer nbc
      real*8 a(im,jm,kb),c(im,jm,kb)
      real*8 ee(im,jm,kb),gg(im,jm,kb)
      real dh(im,jm),rad(im,jm,kb)
      real r(5),ad1(5),ad2(5)
      integer i,j,k,ki

! irradiance parameters after Paulson and Simpson (1977)
!       ntp               1      2       3       4       5
!   Jerlov type           i      ia      ib      ii     iii
      data r   /       .58e0,  .62e0,  .67e0,  .77e0,  .78e0 /
      data ad1 /       .35e0,  .60e0,  1.0e0,  1.5e0,  1.4e0 /
      data ad2 /       23.e0,  20.e0,  17.e0,  14.e0,  7.9e0 /

      double precision proft_time_start_xsz
      double precision proft_time_end_xsz

      call call_time_start(proft_time_start_xsz)
! surface boundary condition:
!       nbc   prescribed    prescribed   short wave
!             temperature      flux      penetration
!             or salinity               (temperature
!                                           only)
!        1        no           yes           no
!        2        no           yes           yes
!        3        yes          no            no
!        4        yes          no            yes
! note that only 1 and 3 are allowed for salinity

! the following section solves the equation
!     dti2*(kh*f')'-f=-fb
*/

/*
void proft_(float f[][j_size][i_size], float wfsurf[][i_size],
		    float fsurf[][i_size], int *f_nbc,
		    float etf[][i_size], float kh[][j_size][i_size],
			float swrad[][i_size]){
*/
/*
void proft_(float f[][j_size][i_size], float wfsurf[][i_size],
		    float fsurf[][i_size], int c_nbc,
		    float etf[][i_size], float kh[][j_size][i_size],
			float swrad[][i_size]){
*/
void proft(float f[][j_size][i_size], 
		   float wfsurf[][i_size],
		   float fsurf[][i_size], 
		   int nbc){

	//modify:
	//		+ f
	//int nbc = *f_nbc;
	//int nbc = c_nbc;
	int i,j,k,ki;
    float a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    float ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
	float dh[j_size][i_size],rad[k_size][j_size][i_size];


	float	r[5]={0.58f,0.62f,0.67f,0.77f,0.78f};
	float	ad1[5]={0.35f,0.60f,1.0f,1.5f,1.4f};
	float	ad2[5]={23.0f,20.0f,17.0f,14.f,7.9f};

/*
! surface boundary condition:
!       nbc   prescribed    prescribed   short wave
!             temperature      flux      penetration
!             or salinity               (temperature
!                                           only)
!        1        no           yes           no
!        2        no           yes           yes
!        3        yes          no            no
!        4        yes          no            yes
! note that only 1 and 3 are allowed for salinity
*/

//we should change kh! for it is changed in profq and reference here

/*
 *etf: re-assigned in advance.f(mode_internal)
 *
 *swrad:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *
 *
 */
/*	
	  do j=1,jm
        do i=1,im
          dh(i,j)=h(i,j)+etf(i,j)
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = h[j][i]+etf[j][i];
		}
	}

/*
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            a(i,j,k-1)=-dti2*(kh(i,j,k)+umol)
     $                  /(dz(k-1)*dzz(k-1)*dh(i,j)*dh(i,j))
            c(i,j,k)=-dti2*(kh(i,j,k)+umol)
     $                  /(dz(k)*dzz(k-1)*dh(i,j)*dh(i,j))
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k-1][j][i] = -dti2*(kh[k][j][i]+umol)
					/(dz[k-1]*dzz[k-1]*dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(kh[k][j][i]+umol)
					/(dz[k]*dzz[k-1]*dh[j][i]*dh[j][i]);

			}
		}
	}	

//! calculate penetrative radiation. At the bottom any unattenuated
//! radiation is deposited in the bottom layer
/*
	do k=1,kb
        do j=1,jm
          do i=1,im
            rad(i,j,k)=0.e0
          end do
        end do
      end do
*/
	for(k = 0; k < kb; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				rad[k][j][i] = 0.0f;
			}
		}
	}
/*
	if(nbc.eq.2.or.nbc.eq.4) then
        do k=1,kbm1
          do j=1,jm
            do i=1,im
              rad(i,j,k)=swrad(i,j)
     $                    *(r(ntp)*exp(z(k)*dh(i,j)/ad1(ntp))
     $                      +(1.e0-r(ntp))*exp(z(k)*dh(i,j)/ad2(ntp)))
            end do
          end do
        end do
      end if
*/
	if(nbc == 2 || nbc == 4 ){
		for(k = 0; k < kbm1; k++){
			for(j = 0; j < jm; j++){
				for(i = 0; i < im; i++){
					rad[k][j][i] = swrad[j][i]*(r[ntp-1]*expf(z[k]*dh[j][i]/ad1[ntp-1])
							+(1.0f-r[ntp-1])*expf(z[k]*dh[j][i]/ad2[ntp-1]));
				}
			}
		}
	}
/*	
      if(nbc.eq.1) then

        do j=1,jm
          do i=1,im
            ee(i,j,1)=a(i,j,1)/(a(i,j,1)-1.e0)
            gg(i,j,1)=-dti2*wfsurf(i,j)/(-dz(1)*dh(i,j))-f(i,j,1)
            gg(i,j,1)=gg(i,j,1)/(a(i,j,1)-1.e0)
          end do
        end do
*/
	if(nbc == 1){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0f);
				gg[0][j][i] = -dti2*wfsurf[j][i]
								/(-dz[0]*dh[j][i])
							  -f[0][j][i];
				gg[0][j][i] = gg[0][j][i]/(a[0][j][i]-1.0f);
			}
		}
	}
/*
      else if(nbc.eq.2) then

        do j=1,jm
          do i=1,im
            ee(i,j,1)=a(i,j,1)/(a(i,j,1)-1.e0)
            gg(i,j,1)=dti2*(wfsurf(i,j)+rad(i,j,1)-rad(i,j,2))
     $                 /(dz(1)*dh(i,j))
     $                   -f(i,j,1)
            gg(i,j,1)=gg(i,j,1)/(a(i,j,1)-1.e0)
          end do
        end do
*/
	else if(nbc == 2){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0f);
				gg[0][j][i] = dti2*(wfsurf[j][i]+rad[0][j][i]-rad[1][j][i])
								/(dz[0]*dh[j][i])
							  -f[0][j][i];
				gg[0][j][i] = gg[0][j][i]/(a[0][j][i]-1.0f);
			}
		}
	}
/*
      else if(nbc.eq.3.or.nbc.eq.4) then

        do j=1,jm
          do i=1,im
            ee(i,j,1)=0.e0
            gg(i,j,1)=fsurf(i,j)
          end do
        end do

      end if
*/
	else if(nbc == 3 || nbc == 4){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				ee[0][j][i]=0.0;
				gg[0][j][i]= fsurf[j][i];
			}
		}
	}
/*
      do k=2,kbm2
        do j=1,jm
          do i=1,im
            gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))-1.e0)
            ee(i,j,k)=a(i,j,k)*gg(i,j,k)
            gg(i,j,k)=(c(i,j,k)*gg(i,j,k-1)-f(i,j,k)
     $                 +dti2*(rad(i,j,k)-rad(i,j,k+1))
     $                   /(dh(i,j)*dz(k)))
     $                 *gg(i,j,k)
          end do
        end do
      end do
*/
	for(k = 1; k < kbm2; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0f/(a[k][j][i]
								    +c[k][j][i]
										*(1.0f-ee[k-1][j][i])
									-1.0f);
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (c[k][j][i]*gg[k-1][j][i]
								-f[k][j][i]
								+dti2*(rad[k][j][i]-rad[k+1][j][i])
									/(dh[j][i]*dz[k]))
							   *gg[k][j][i];
			}
		}
	}
/*
! bottom adiabatic boundary condition
      do j=1,jm
        do i=1,im
          f(i,j,kbm1)=(c(i,j,kbm1)*gg(i,j,kbm2)-f(i,j,kbm1)
     $                 +dti2*(rad(i,j,kbm1)-rad(i,j,kb))
     $                   /(dh(i,j)*dz(kbm1)))
     $                 /(c(i,j,kbm1)*(1.e0-ee(i,j,kbm2))-1.e0)
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			f[kbm1-1][j][i] = (c[kbm1-1][j][i]*gg[kbm2-1][j][i]
								-f[kbm1-1][j][i]
								+dti2*(rad[kbm1-1][j][i]
									  -rad[kb-1][j][i])
								  /(dh[j][i]*dz[kbm1-1]))
							  /(c[kbm1-1][j][i]
								*(1.0f-ee[kbm2-1][j][i])
							   -1.0f);	
		}
	}
/*
      do k=2,kbm1
        ki=kb-k
        do j=1,jm
          do i=1,im
            f(i,j,ki)=(ee(i,j,ki)*f(i,j,ki+1)+gg(i,j,ki))
          end do
        end do
      end do
*/
	for(ki = kb-3; ki >= 0; ki--){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				f[ki][j][i] = (ee[ki][j][i]*f[ki+1][j][i]+gg[ki][j][i]);
			}
		}
	}
    return;
}


/*
subroutine advu
! do horizontal and vertical advection of u-momentum, and includes
! coriolis, surface slope and baroclinic terms
      implicit none
      include 'pom.h'
      integer i,j,k

      double precision advu_time_start_xsz
      double precision advu_time_end_xsz

      call call_time_start(advu_time_start_xsz)
*/

/*
void advu_(float uf[][j_size][i_size], float w[][j_size][i_size],
		  float u[][j_size][i_size], float advx[][j_size][i_size],
		  float dt[][i_size], float v[][j_size][i_size],
		  float egf[][i_size], float egb[][i_size],
		  float e_atmos[][i_size], float drhox[][j_size][i_size],
		  float etb[][i_size], float ub[][j_size][i_size],
		  float etf[][i_size]){
*/

void advu(){

	int i,j,k;
	//modify: - uf
/*
 *uf: read in advance.f
 *
 *u: read and write in advance.f
 *
 *w: assigned in advance.f (surface)
 *
 *advx: read in advance.f 
 *
 *dt: write again in advance.f dt = h + et
 *
 *v: read and write in advance.f
 *
 *egf: read in advance.f
 *
 *egb: assigned in advance.f egb=egf
 *
 *drhox: write in other parts of solver.f 
 *
 *e_atmos:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *etb: re-assigned in advance.f(mode_internal)
 *
 *ub: read and write in advance.f
 *
 *etf: re-assigned in advance.f(mode_internal)
 */
//! do vertical advection
/*
      do k=1,kb
        do j=1,jm
          do i=1,im
            uf(i,j,k)=0.e0
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uf[k][j][i] = 0.0f;	
			}
		}
	}

/*
      do k=2,kbm1
        do j=1,jm
          do i=2,im
            uf(i,j,k)=.25e0*(w(i,j,k)+w(i-1,j,k))
     $                     *(u(i,j,k)+u(i,j,k-1))
          end do
        end do
      end do
*/
	for (k = 1; k < kbm1; k++){
		for (j = 0; j < jm; j++){
			for (i = 1; i < im; i++){
				uf[k][j][i]=0.25f*(w[k][j][i]+w[k][j][i-1])*(u[k][j][i]+u[k-1][j][i]);
			}
		}
	}


//! combine horizontal and vertical advection with coriolis, surface
//! slope and baroclinic terms
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            uf(i,j,k)=advx(i,j,k)
     $                 +(uf(i,j,k)-uf(i,j,k+1))*aru(i,j)/dz(k)
     $                 -aru(i,j)*.25e0
     $                   *(cor(i,j)*dt(i,j)
     $                      *(v(i,j+1,k)+v(i,j,k))
     $                     +cor(i-1,j)*dt(i-1,j)
     $                       *(v(i-1,j+1,k)+v(i-1,j,k)))
     $                 +grav*.125e0*(dt(i,j)+dt(i-1,j))
     $                   *(egf(i,j)-egf(i-1,j)+egb(i,j)-egb(i-1,j)
     $                     +(e_atmos(i,j)-e_atmos(i-1,j))*2.e0)
     $                   *(dy(i,j)+dy(i-1,j))
     $                 +drhox(i,j,k)
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				uf[k][j][i]=advx[k][j][i]
					         +(uf[k][j][i]-uf[k+1][j][i])*aru[j][i]/dz[k]
							 -aru[j][i]*0.25f
								*(cor[j][i]*dt[j][i]*(v[k][j+1][i]+v[k][j][i])
									  +cor[j][i-1]*dt[j][i-1]*(v[k][j+1][i-1]+v[k][j][i-1]))
							 +grav*0.125f*(dt[j][i]+dt[j][i-1])
								*(egf[j][i]-egf[j][i-1]+egb[j][i]-egb[j][i-1]
										+(e_atmos[j][i]-e_atmos[j][i-1])*2.0f)
								*(dy[j][i]+dy[j][i-1])
							 +drhox[k][j][i];
			}
		}
	}

//!  step forward in time
/*
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            uf(i,j,k)=((h(i,j)+etb(i,j)+h(i-1,j)+etb(i-1,j))
     $                 *aru(i,j)*ub(i,j,k)
     $                 -2.e0*dti2*uf(i,j,k))
     $                /((h(i,j)+etf(i,j)+h(i-1,j)+etf(i-1,j))
     $                  *aru(i,j))
          end do
        end do
      end do
*/
	for (k = 0; k < kbm1; k++){
		for (j = 1; j < jmm1; j++){
			for (i = 1; i < imm1; i++){
				uf[k][j][i]=((h[j][i]+etb[j][i]+h[j][i-1]+etb[j][i-1])
								*aru[j][i]*ub[k][j][i]
								-2.0f*dti2*uf[k][j][i])
							/((h[j][i]+etf[j][i]+h[j][i-1]+etf[j][i-1])*aru[j][i]);
			}	
		}
	}

	return;
}


/*
      subroutine advv
! do horizontal and vertical advection of v-momentum, and includes
! coriolis, surface slope and baroclinic terms
      implicit none
      include 'pom.h'
      integer i,j,k

      double precision advv_time_start_xsz
      double precision advv_time_end_xsz

      call call_time_start(advv_time_start_xsz)
*/
/*
void advv_(float vf[][j_size][i_size], float w[][j_size][i_size],
		  float u[][j_size][i_size], float advy[][j_size][i_size],
		  float dt[][i_size], float v[][j_size][i_size],
		  float egf[][i_size], float egb[][i_size],
		  float e_atmos[][i_size], float drhoy[][j_size][i_size],
		  float etb[][i_size], float vb[][j_size][i_size],
		  float etf[][i_size]){
*/
void advv(){

	int i,j,k;
/*
 *vf: read in advance.f
 *
 *u: read and write in advance.f
 *
 *w: assigned in advance.f (surface)
 *
 *advy: read in advance.f 
 *
 *dt: write again in advance.f dt = h + et
 *
 *v: read and write in advance.f
 *
 *egf: read in advance.f
 *
 *egb: assigned in advance.f egb=egf
 *
 *drhoy: write in other parts of solver.f 
 *
 *e_atmos:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *etb: re-assigned in advance.f(mode_internal)
 *
 *vb: read and write in advance.f
 *
 *etf: re-assigned in advance.f(mode_internal)
 */

//! do vertical advection
/*     
      do k=1,kb
        do j=1,jm
          do i=1,im
            vf(i,j,k)=0.e0
          end do
        end do
      end do
*/
	for (k = 0; k < kb; k++){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				vf[k][j][i] = 0.0f;
			}
		}
	}
/*     
      do k=2,kbm1
        do j=2,jm
          do i=1,im
            vf(i,j,k)=.25e0*(w(i,j,k)+w(i,j-1,k))
     $                     *(v(i,j,k)+v(i,j,k-1))
          end do
        end do
      end do
*/
	for (k = 1; k < kbm1; k++){
		for(j = 1; j < jm; j++){
			for(i = 0; i < im; i++){
				vf[k][j][i] = 0.25f*(w[k][j][i]+w[k][j-1][i])*(v[k][j][i]+v[k-1][j][i]);
			}
		}
	}
	
//! combine horizontal and vertical advection with coriolis, surface
//! slope and baroclinic terms
/*  
   do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            vf(i,j,k)=advy(i,j,k)
     $                 +(vf(i,j,k)-vf(i,j,k+1))*arv(i,j)/dz(k)
     $                 +arv(i,j)*.25e0
     $                   *(cor(i,j)*dt(i,j)
     $                      *(u(i+1,j,k)+u(i,j,k))
     $                     +cor(i,j-1)*dt(i,j-1)
     $                       *(u(i+1,j-1,k)+u(i,j-1,k)))
     $                 +grav*.125e0*(dt(i,j)+dt(i,j-1))
     $                   *(egf(i,j)-egf(i,j-1)+egb(i,j)-egb(i,j-1)
     $                     +(e_atmos(i,j)-e_atmos(i,j-1))*2.e0)
     $                   *(dx(i,j)+dx(i,j-1))
     $                 +drhoy(i,j,k)
          end do
        end do
      end do
*/
	for(k = 0; k < kbm1; k++){
		for(j = 1; j <jmm1; j++){
			for(i = 1; i <imm1; i++){
				vf[k][j][i] = advy[k][j][i]
								+(vf[k][j][i]-vf[k+1][j][i])*arv[j][i]/dz[k]
								+arv[j][i]*0.25f
									*(cor[j][i]*dt[j][i]*(u[k][j][i+1]+u[k][j][i])
										+cor[j-1][i]*dt[j-1][i]*(u[k][j-1][i+1]+u[k][j-1][i]))
								+grav*0.125f*(dt[j][i]+dt[j-1][i])
									*(egf[j][i]-egf[j-1][i]+egb[j][i]-egb[j-1][i]
										+(e_atmos[j][i]-e_atmos[j-1][i])*2.0f)
									*(dx[j][i]+dx[j-1][i])
								+drhoy[k][j][i];
				}
			}
		}
		
//! step forward in time
/*      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            vf(i,j,k)=((h(i,j)+etb(i,j)+h(i,j-1)+etb(i,j-1))
     $                 *arv(i,j)*vb(i,j,k)
     $                 -2.e0*dti2*vf(i,j,k))
     $                /((h(i,j)+etf(i,j)+h(i,j-1)+etf(i,j-1))
     $                  *arv(i,j))
          end do
        end do
      end do
*/
	for(k = 0; k < kbm1; k++){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				vf[k][j][i] = ((h[j][i]+etb[j][i]+h[j-1][i]+etb[j-1][i])
									*arv[j][i]*vb[k][j][i]
							-2.0f*dti2*vf[k][j][i])
								/((h[j][i]+etf[j][i]+h[j-1][i]+etf[j-1][i])
									*arv[j][i]);
			}
		}
	}
/*
      call call_time_end(advv_time_end_xsz)
      advv_time_xsz = advv_time_end_xsz - advv_time_start_xsz
     $              + advv_time_xsz
*/
      return;
}


/*
!_______________________________________________________________________
      subroutine profu
! solves for vertical diffusion of x-momentum using method described by
! Richmeyer and Morton (1967)
! note: wusurf has the opposite sign to the wind speed
      implicit none
      include 'pom.h'
      real*8 a(im,jm,kb),c(im,jm,kb)
      real*8 ee(im,jm,kb),gg(im,jm,kb)
      real dh(im,jm)
      integer i,j,k,ki

      double precision profu_time_start_xsz
      double precision profu_time_end_xsz

      call call_time_start(profu_time_start_xsz)

! the following section solves the equation
!   dti2*(km*u')'-u=-ub
*/

/*
void profu_(float etf[][i_size],float km[][j_size][i_size],
			float wusurf[][i_size],float uf[][j_size][i_size],
			float vb[][j_size][i_size],float ub[][j_size][i_size],
			float wubot[][i_size]){
*/
void profu(){

	int i,j,k,ki;
    float a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    float ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
	float dh[j_size][i_size];
	float tps[j_size][i_size];

//modify +uf -wubot
//we should change km! for it is changed here and reference only in profv & profu
//we should change wubot! for it is changed in advave_ only! and referenced here and profu
/*
 *etf: re-assigned in advance.f(mode_internal)
 *
 * tps : read and write in advance.f
 *       but in advance.f tps is always assigned a value before
 *       so I believe it is a tmp array; do not reference in
 *
 *wusurf:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *uf: read in advance.f
 *
 *vb: read and write in advance.f
 *
 *ub: read and write in advance.f
 *
 */

/*
      do j=1,jm
        do i=1,im
          dh(i,j)=1.e0
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = 1.0f;
		}
	}
/*
      do j=2,jm
        do i=2,im
          dh(i,j)=(h(i,j)+etf(i,j)+h(i-1,j)+etf(i-1,j))*.5e0
        end do
      end do
*/

	for(j = 1; j < jm; j++){
		for(i = 1; i < im; i++){
			dh[j][i] = (h[j][i]+etf[j][i]+h[j][i-1]+etf[j][i-1])*0.5f;
		}
	}

/*
      do k=1,kb
        do j=2,jm
          do i=2,im
            c(i,j,k)=(km(i,j,k)+km(i-1,j,k))*.5e0
          end do
        end do
      end do
*/
	for(k = 0; k < kb; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				c[k][j][i] = (km[k][j][i]+km[k][j][i-1])*0.5f;
			}
		}
	}
/*
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            a(i,j,k-1)=-dti2*(c(i,j,k)+umol)
     $                  /(dz(k-1)*dzz(k-1)*dh(i,j)*dh(i,j))
            c(i,j,k)=-dti2*(c(i,j,k)+umol)
     $                /(dz(k)*dzz(k-1)*dh(i,j)*dh(i,j))
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k-1][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k-1]*dzz[k-1]*dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k]*dzz[k-1]*dh[j][i]*dh[j][i]);
			}
		}
	}
/*
      do j=1,jm
        do i=1,im
          ee(i,j,1)=a(i,j,1)/(a(i,j,1)-1.e0)
          gg(i,j,1)=(-dti2*wusurf(i,j)/(-dz(1)*dh(i,j))
     $               -uf(i,j,1))
     $               /(a(i,j,1)-1.e0)
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0f);
			gg[0][j][i] = (-dti2*wusurf[j][i]/(-dz[0]*dh[j][i])
							-uf[0][j][i])/(a[0][j][i]-1.0f);
		}
	}
/*	
      do k=2,kbm2
        do j=1,jm
          do i=1,im
            gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))-1.e0)
            ee(i,j,k)=a(i,j,k)*gg(i,j,k)
            gg(i,j,k)=(c(i,j,k)*gg(i,j,k-1)-uf(i,j,k))*gg(i,j,k)
          end do
        end do
      end do
*/
	for(k = 1; k < kbm2; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0f/(a[k][j][i]
									 +c[k][j][i]
										*(1.0f-ee[k-1][j][i])
									-1.0f);
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (c[k][j][i]*gg[k-1][j][i]
								-uf[k][j][i])
							  *gg[k][j][i];
			}
		}
	}
/*	
      do j=2,jmm1
        do i=2,imm1
          tps(i,j)=0.5e0*(cbc(i,j)+cbc(i-1,j))
     $              *sqrt(ub(i,j,kbm1)**2
     $                +(.25e0*(vb(i,j,kbm1)+vb(i,j+1,kbm1)
     $                         +vb(i-1,j,kbm1)+vb(i-1,j+1,kbm1)))**2)
          uf(i,j,kbm1)=(c(i,j,kbm1)*gg(i,j,kbm2)-uf(i,j,kbm1))
     $                  /(tps(i,j)*dti2/(-dz(kbm1)*dh(i,j))-1.e0
     $                    -(ee(i,j,kbm2)-1.e0)*c(i,j,kbm1))
          uf(i,j,kbm1)=uf(i,j,kbm1)*dum(i,j)
        end do
      end do
*/
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			float tmp = 0.25f*(vb[kbm1-1][j][i]
							  +vb[kbm1-1][j+1][i]
							  +vb[kbm1-1][j][i-1]
							  +vb[kbm1-1][j+1][i-1]);
			/*
			tps[j][i] = 0.5f*(cbc[j][i]+cbc[j][i-1])
						 *sqrtf(ub[kbm1-1][j][i]*ub[kbm1-1][j][i]+powf((0.25f*(vb[kbm1-1][j][i]+vb[kbm1-1][j+1][i]+vb[kbm1-1][j][i-1]+vb[kbm1-1][j+1][i-1]),2)));
			*/
			tps[j][i] = 0.5f*(cbc[j][i]+cbc[j][i-1])
						 *sqrtf((ub[kbm1-1][j][i]
								 *ub[kbm1-1][j][i])
								+(tmp*tmp));

			uf[kbm1-1][j][i] = (c[kbm1-1][j][i]*gg[kbm2-1][j][i]
									-uf[kbm1-1][j][i])
								/(tps[j][i]*dti2
									/(-dz[kbm1-1]*dh[j][i])
								  -1.0f
						          -(ee[kbm2-1][j][i]-1.0f)
									*c[kbm1-1][j][i]);

			uf[kbm1-1][j][i] = uf[kbm1-1][j][i]*dum[j][i];
		}
	}
/*
      do k=2,kbm1
        ki=kb-k
        do j=2,jmm1
          do i=2,imm1
            uf(i,j,ki)=(ee(i,j,ki)*uf(i,j,ki+1)+gg(i,j,ki))*dum(i,j)
          end do
        end do
      end do
*/
	for(ki = kb-3; ki >= 0; ki--){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				uf[ki][j][i] = (ee[ki][j][i]*uf[ki+1][j][i]+gg[ki][j][i])*dum[j][i];
			}
		}
	}

/*
      do j=2,jmm1
        do i=2,imm1
          wubot(i,j)=-tps(i,j)*uf(i,j,kbm1)
        end do
      end do
*/
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			wubot[j][i] = -tps[j][i]*uf[kbm1-1][j][i];
		}
	}
    //exchange2d_mpi(wubot,im,jm)
	
    //exchange2d_mpi_xsz_(wubot,im,jm);
    exchange2d_mpi(wubot,im,jm);
    return;
}



/*
!_______________________________________________________________________
      subroutine profv
! solves for vertical diffusion of x-momentum using method described by
! Richmeyer and Morton (1967)
! note: wvsurf has the opposite sign to the wind speed
      implicit none
      include 'pom.h'
      real*8 a(im,jm,kb),c(im,jm,kb)
      real*8 ee(im,jm,kb),gg(im,jm,kb)
      real dh(im,jm)
      integer i,j,k,ki

      double precision profv_time_start_xsz
      double precision profv_time_end_xsz

      call call_time_start(profv_time_start_xsz)

! the following section solves the equation
!     dti2*(km*u')'-u=-ub
*/

/*
void profv_(float etf[][i_size],float km[][j_size][i_size],
			float wvsurf[][i_size],float vf[][j_size][i_size],
			float ub[][j_size][i_size],float vb[][j_size][i_size],
			 float wvbot[][i_size]){
*/
void profv(){

	int i,j,k,ki;
    float a[k_size][j_size][i_size],c[k_size][j_size][i_size];
    float ee[k_size][j_size][i_size],gg[k_size][j_size][i_size];
	float dh[j_size][i_size];
	float tps[j_size][i_size];

//we should change km! for it is changed here and reference only in profv & profu
//we should change wvbot! for it is changed in profv_ only!
/*
 *etf: re-assigned in advance.f(mode_internal)
 *
 *wvsurf:although it is always 0 assigned in advance.f(surface)
 *        for the development of future, it should be reference for none-0 possiblity
 *
 *vf: read in advance.f
 *
 *ub: read and write in advance.f
 *
 *vb: read and write in advance.f
 */
/*
      do j=1,jm
        do i=1,im
          dh(i,j)=1.e0
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			dh[j][i] = 1.0f;
		}
	}
/*	
      do j=2,jm
        do i=2,im
          dh(i,j)=.5e0*(h(i,j)+etf(i,j)+h(i,j-1)+etf(i,j-1))
        end do
      end do
*/
	for(j = 1; j < jm; j++){
		for(i = 1; i < im; i++){
			dh[j][i] = 0.5f*(h[j][i]+etf[j][i]+h[j-1][i]+etf[j-1][i]);
		}
	}

/*
      do k=1,kb
        do j=2,jm
          do i=2,im
            c(i,j,k)=(km(i,j,k)+km(i,j-1,k))*.5e0
          end do
        end do
      end do
*/
	for(k = 0; k < kb; k++){
		for(j = 1; j < jm; j++){
			for(i = 1; i < im; i++){
				c[k][j][i] = (km[k][j][i]+km[k][j-1][i])*0.5f;
			}
		}
	}
/*	
      do k=2,kbm1
        do j=1,jm
          do i=1,im
            a(i,j,k-1)=-dti2*(c(i,j,k)+umol)
     $                  /(dz(k-1)*dzz(k-1)*dh(i,j)*dh(i,j))
            c(i,j,k)=-dti2*(c(i,j,k)+umol)
     $                /(dz(k)*dzz(k-1)*dh(i,j)*dh(i,j))
          end do
        end do
      end do
*/
	for(k = 1; k < kbm1; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				a[k-1][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k-1]*dzz[k-1]*dh[j][i]*dh[j][i]);
				c[k][j][i] = -dti2*(c[k][j][i]+umol)
					/(dz[k]*dzz[k-1]*dh[j][i]*dh[j][i]);
			}
		}
	}
/*
      do j=1,jm
        do i=1,im
          ee(i,j,1)=a(i,j,1)/(a(i,j,1)-1.e0)
          gg(i,j,1)=(-dti2*wvsurf(i,j)/(-dz(1)*dh(i,j))-vf(i,j,1))
     $               /(a(i,j,1)-1.e0)
        end do
      end do
*/
	for(j = 0; j < jm; j++){
		for(i = 0; i < im; i++){
			ee[0][j][i] = a[0][j][i]/(a[0][j][i]-1.0f);
			gg[0][j][i] = (-dti2*wvsurf[j][i]
							/(-dz[0]*dh[j][i])
						   -vf[0][j][i])
						 /(a[0][j][i]-1.0f);
		}
	}
/*
      do k=2,kbm2
        do j=1,jm
          do i=1,im
            gg(i,j,k)=1.e0/(a(i,j,k)+c(i,j,k)*(1.e0-ee(i,j,k-1))-1.e0)
            ee(i,j,k)=a(i,j,k)*gg(i,j,k)
            gg(i,j,k)=(c(i,j,k)*gg(i,j,k-1)-vf(i,j,k))*gg(i,j,k)
          end do
        end do
      end do
*/
	for(k = 1; k < kbm2; k++){
		for(j = 0; j < jm; j++){
			for(i = 0; i < im; i++){
				gg[k][j][i] = 1.0f/(a[k][j][i]
									 +c[k][j][i]*(1.0f-ee[k-1][j][i])
									-1.0f);
				ee[k][j][i] = a[k][j][i]*gg[k][j][i];
				gg[k][j][i] = (c[k][j][i]*gg[k-1][j][i]
								-vf[k][j][i])
							  *gg[k][j][i];
			}
		}
	}
/*
      do j=2,jmm1
        do i=2,imm1
          tps(i,j)=0.5e0*(cbc(i,j)+cbc(i,j-1))
     $              *sqrt((.25e0*(ub(i,j,kbm1)+ub(i+1,j,kbm1)
     $                            +ub(i,j-1,kbm1)+ub(i+1,j-1,kbm1)))**2
     $                    +vb(i,j,kbm1)**2)
          vf(i,j,kbm1)=(c(i,j,kbm1)*gg(i,j,kbm2)-vf(i,j,kbm1))
     $                  /(tps(i,j)*dti2/(-dz(kbm1)*dh(i,j))-1.e0
     $                    -(ee(i,j,kbm2)-1.e0)*c(i,j,kbm1))
          vf(i,j,kbm1)=vf(i,j,kbm1)*dvm(i,j)
        end do
      end do
*/
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			float tmp = 0.25f*(ub[kbm1-1][j][i]
							  +ub[kbm1-1][j][i+1]
							  +ub[kbm1-1][j-1][i]
							  +ub[kbm1-1][j-1][i+1]);

			tps[j][i] = 0.5f*(cbc[j][i]+cbc[j-1][i])
							*sqrtf((tmp*tmp)
								  +(vb[kbm1-1][j][i]
									  *vb[kbm1-1][j][i]));

			vf[kbm1-1][j][i] = (c[kbm1-1][j][i]*gg[kbm2-1][j][i]
								-vf[kbm1-1][j][i])
							  /(tps[j][i]*dti2/(-dz[kbm1-1]*dh[j][i])
								-1.0f
								-(ee[kbm2-1][j][i]-1.0f)
									*c[kbm1-1][j][i]);

			vf[kbm1-1][j][i] = vf[kbm1-1][j][i]*dvm[j][i];
		}
	}
/*
      do k=2,kbm1
        ki=kb-k
        do j=2,jmm1
          do i=2,imm1
            vf(i,j,ki)=(ee(i,j,ki)*vf(i,j,ki+1)+gg(i,j,ki))*dvm(i,j)
          end do
        end do
      end do
*/
	for(ki = kb-3; ki >= 0; ki--){
		for(j = 1; j < jmm1; j++){
			for(i = 1; i < imm1; i++){
				vf[ki][j][i] = (ee[ki][j][i]*vf[ki+1][j][i]
									+gg[ki][j][i])
							   *dvm[j][i];
			}
		}
	}
/*
      do j=2,jmm1
        do i=2,imm1
          wvbot(i,j)=-tps(i,j)*vf(i,j,kbm1)
        end do
      end do
*/
	for(j = 1; j < jmm1; j++){
		for(i = 1; i < imm1; i++){
			wvbot[j][i] = -tps[j][i]*vf[kbm1-1][j][i];
		}
	}

    //call exchange2d_mpi(wvbot,im,jm)
    //exchange2d_mpi_xsz_(wvbot,im,jm);
    exchange2d_mpi(wvbot,im,jm);
      
	return;
}
