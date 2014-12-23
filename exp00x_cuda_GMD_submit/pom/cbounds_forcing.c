#include"cbounds_forcing.h"
// bounds_forcing.f

// spcify variable boundary conditions, atmospheric forcing, restoring


/*
!     Below are (4)  bcond* for various applicaitons:
!     The actual one used by the model is subroutine bcond
!     The others are ignored so must change name (to bcond) to use
!
!_______________________________________________________________________
      subroutine bcond(idx)
!     bcond_PeriodicFRZ.f:
!     boundary conditions for idelaized STCC simulation with 
!     x-periodic and y-FRZ;
!     need also subroutines xperi2d_mpi & xperi3d_mpi
!     included in this are also yperi*:ipery: for y-periodic
!
      implicit none
      include 'pom.h'
      integer idx
      integer i,j,k
      real ga,u1,wm
      integer ii,jj
      real, parameter :: hmax = 8000.0
*/

void bcond(int idx){
	int i, j, k;
	float ga, u1, wm;
	const float hmax = 8000.f;
      
/*
      if(idx.eq.1) then

! external (2-D) elevation boundary conditions

!west
          if(n_west.eq.-1) then
           do j=1,jm
             elf(1,j)=elf(2,j)
           end do
          endif
!east
          if(n_east.eq.-1) then 
           do j=1,jm
             elf(im,j)=elf(imm1,j)
           end do
          endif
!north
          if(n_north.eq.-1) then
           do i=1,im
            elf(i,jm)=elf(i,jmm1)
           end do
          endif
!south
          if(n_south.eq.-1) then
           do i=1,im
            elf(i,1)=elf(i,2)
           end do
          endif
!
        if (iperx.ne.0) then
         call xperi2d_mpi(elf,im,jm)
         endif
!
        if (ipery.ne.0) then 
         call yperi2d_mpi(elf,im,jm)
         endif
!
          do j=1,jm
             do i=1,im
                elf(i,j)=elf(i,j)*fsm(i,j)
             end do
          end do

        return
*/
	if (idx == 1){

		if (n_west == -1){
			for (j = 0; j < jm; j++){
				elf[j][0] = elf[j][1];	
			}
		}
		if (n_east == -1){
			for (j = 0; j < jm; j++){
				elf[j][im-1] = elf[j][imm1-1];	
			}
		}
		if (n_north == -1){
			for (i = 0; i < im; i++){
				elf[jm-1][i] = elf[jmm1-1][i];	
			}
		}
		if (n_south == -1){
			for (i = 0; i < im; i++){
				elf[0][i] = elf[1][i];	
			}
		}

		if (iperx != 0){
			xperi2d_mpi(elf, im, jm);
		}

		if (ipery != 0){
			yperi2d_mpi(elf, im, jm);	
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				elf[j][i] = elf[j][i]*fsm[j][i];	
			}
		}

	}

/*
      else if(idx.eq.2) then

! external (2-D) velocity boundary conditions
        
! east
          if(n_east.eq.-1) then
           do j=2,jmm1
            uaf(im,j)=uabe(j)
     $                     +rfe*sqrt(grav/h(imm1,j))*(el(imm1,j)-ele(j))
            uaf(im,j)=ramp*uaf(im,j)
            vaf(im,j)=0.e0
           enddo
          end if
! west
          if(n_west.eq.-1) then
           do j=2,jmm1
            uaf(2,j)=uabw(j)-rfw*sqrt(grav/h(2,j))*(el(2,j)-elw(j))
            uaf(2,j)=ramp*uaf(2,j)
            uaf(1,j)=uaf(2,j)
            vaf(1,j)=0.e0
           enddo
          end if

! north
          if(n_north.eq.-1) then
           do i=2,imm1
            vaf(i,jm)=vabn(i)
     $                     +rfn*sqrt(grav/h(i,jmm1))*(el(i,jmm1)-eln(i))
            vaf(i,jm)=ramp*vaf(i,jm)
            uaf(i,jm)=0.e0
           enddo
          end if
! south
          if(n_south.eq.-1) then
           do i=2,imm1
            vaf(i,2)=vabs(i)-rfs*sqrt(grav/h(i,2))*(el(i,2)-els(i))
            vaf(i,2)=ramp*vaf(i,2)
            vaf(i,1)=vaf(i,2)
            uaf(i,1)=0.e0
           enddo
          end if
!
        if (iperx.ne.0) then
          call xperi2d_mpi(uaf,im,jm)
          call xperi2d_mpi(vaf,im,jm)
        if (iperx.lt.0) then
           if(n_north.eq.-1) then 
             uaf(:,jm)=uaf(:,jmm1)
             dum(:,jm)=1.0
           endif
           if(n_south.eq.-1) then 
              uaf(:,1)=uaf(:,2)
              dum(:,1)=1.0
           endif
         endif
        endif !end if (iperx.ne.0)
!
        if (ipery.ne.0) then
          call yperi2d_mpi(uaf,im,jm)
          call yperi2d_mpi(vaf,im,jm)
        if (ipery.lt.0) then !free-slip east&west
           if(n_east.eq.-1) then 
             vaf(im,:)=vaf(imm1,:)
             dvm(im,:)=1.0
           endif
           if(n_west.eq.-1) then 
              vaf(1,:)=vaf(2,:)
              dvm(1,:)=1.0
           endif
         endif
        endif !end if (ipery.ne.0)
!
         do j=1,jm
            do i=1,im
               uaf(i,j)=uaf(i,j)*dum(i,j)
               vaf(i,j)=vaf(i,j)*dvm(i,j)
            end do
         end do

        return
*/
	else if (idx == 2){
		if (n_east == -1){
			for (j = 1; j < jmm1; j++){
				uaf[j][im-1] = uabe[j]+rfe
									  *sqrtf(grav/h[j][imm1-1])
									  *(el[j][imm1-1]-ele[j]);
				uaf[j][im-1] = ramp*uaf[j][im-1];
				vaf[j][im-1] = 0;

			}
		}
		if (n_west == -1){
			for (j = 1; j < jmm1; j++){
				uaf[j][1] = uabw[j]-rfw*sqrtf(grav/h[j][1])
									   *(el[j][1]-elw[j]);		
				uaf[j][1] = ramp*uaf[j][1];
				uaf[j][0] = uaf[j][1];
				vaf[j][0] = 0;
			}
		}

		if (n_north == -1){
			for (i = 1; i < imm1; i++){
				vaf[jm-1][i] = vabn[i]+rfn
									  *sqrtf(grav/h[jmm1-1][i])	
									  *(el[jmm1-1][i]-eln[i]);
				vaf[jm-1][i] = ramp*vaf[jm-1][i];
				uaf[jm-1][i] = 0;
			}
		}

		if (n_south == -1){
			for (i = 1; i < imm1; i++){
				vaf[1][i] = vabs[i]-rfs*sqrtf(grav/h[1][i])
									   *(el[1][i]-els[i]);
				vaf[1][i] = ramp*vaf[1][i];
				vaf[0][i] = vaf[1][i];
				uaf[0][i] = 0;
			}
		}

		if (iperx != 0){
			xperi2d_mpi(uaf, im, jm);	
			xperi2d_mpi(vaf, im, jm);	
			if (iperx < 0){
				if (n_north == -1){
					for (i = 0; i < im; i++){
						uaf[jm-1][i] = uaf[jmm1-1][i];
						dum[jm-1][i] = 1.f;
					}
				}
				if (n_south == -1){
					for (i = 0; i < im; i++){
						uaf[0][i] = uaf[1][i];
						dum[0][i] = 1.f;
					}
				}
			}
		}

		if (ipery != 0){
			yperi2d_mpi(uaf, im, jm);	
			yperi2d_mpi(vaf, im, jm);	
			if (ipery < 0){
				if (n_east == -1){
					for (j = 0; j < jm; j++){
						vaf[j][im-1] = vaf[j][imm1-1];	
						dvm[j][im-1] = 1.f;
					}
				}

				if (n_west == -1){
					for (j = 0; j < jm; j++){
						vaf[j][0] = vaf[j][1];
						dvm[j][0] = 1.f;
					}
				}
			}
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uaf[j][i] = uaf[j][i]*dum[j][i];
				vaf[j][i] = vaf[j][i]*dvm[j][i];
			}
		}
	}

/*

      else if(idx.eq.3) then

! internal (3-D) velocity boundary conditions

!     EAST
!     radiation boundary conditions.

         if(n_east.eq.-1) then                  
            do k=1,kbm1
               do j=2,jmm1
                  ga = sqrt( h(im,j) / hmax )    
                  uf(im,j,k)  
     $                 = ga * ( 0.25 * u(imm1,j-1,k) 
     $                 + 0.5 * u(imm1,j,k) + 0.25 * u(imm1,j+1,k) )
     $                 + ( 1.0 - ga ) * ( 0.25 * u(im,j-1,k) 
     $                 + 0.5 * u(im,j,k) + 0.25 * u(im,j+1,k) )
                  vf(im,j,k)=0.e0
               enddo
            enddo
         endif

!     WEST
!     radiation boundary conditions.

         if(n_west.eq.-1) then                  
            do k=1,kbm1
               do j=2,jmm1
                  ga = sqrt( h(1,j) / hmax )
                  uf(2,j,k)  
     $                 = ga * ( 0.25 * u(3,j-1,k) 
     $                 + 0.5 * u(3,j,k) + 0.25 * u(3,j+1,k) )
     $                 + ( 1.0 - ga ) * ( 0.25 * u(2,j-1,k) 
     $                 + 0.5 * u(2,j,k) + 0.25 * u(2,j+1,k) )
                  uf(1,j,k)=uf(2,j,k)
                  vf(1,j,k)=0.e0
               enddo
            enddo
          endif

!     NORTH
!     radiation boundary conditions.

         if(n_north.eq.-1) then                  
            
            do k=1,kbm1
               do i=2,imm1
                  ga = sqrt( h(i,jm) / hmax )
                  vf(i,jm,k)  
     $                 = ga * ( 0.25 * v(i-1,jmm1,k) 
     $                 + 0.5 * v(i,jmm1,k) + 0.25 * v(i+1,jmm1,k) )
     $                 + ( 1.0 - ga ) * ( 0.25 * v(i-1,jm,k) 
     $                 + 0.5 * v(i,jm,k) + 0.25 * v(i+1,jm,k) )
                  uf(i,jm,k)=0.e0
               enddo
            enddo
          endif

!     SOUTH
!     radiation boundary conditions.

         if(n_south.eq.-1) then                  
            
            do k=1,kbm1
               do i=2,imm1
                  ga = sqrt( h(i,1) / hmax )
                  vf(i,2,k)  
     $                 = ga * ( 0.25 * v(i-1,3,k) 
     $                 + 0.5 * v(i,3,k) + 0.25 * v(i+1,3,k) )
     $                 + ( 1.0 - ga ) * ( 0.25 * v(i-1,2,k) 
     $                 + 0.5 * v(i,2,k) + 0.25 * v(i+1,2,k) )
                  vf(i,1,k)=vf(i,2,k)
                  uf(i, 1,k)=0.e0
               enddo
            enddo
          endif
!
        if (iperx.ne.0) then
        call xperi2d_mpi(wubot,im,jm)
        call xperi2d_mpi(wvbot,im,jm)
        call xperi3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
        call xperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)

        if (iperx.lt.0) then !free-slip north&south
           if(n_north.eq.-1)           then 
              wubot(:,jm)=wubot(:,jmm1)
              do k=1,kbm1
               uf(:,jm,k)=uf(:,jmm1,k)
              enddo
            endif
           if(n_south.eq.-1)           then
              wubot(:,1)=wubot(:,2)        
              do k=1,kbm1
               uf(:,1,k)=uf(:,2,k)
              enddo
            endif
         endif
        endif ! end if (iperx.ne.0)

        if (ipery.ne.0) then
        call yperi2d_mpi(wubot,im,jm)
        call yperi2d_mpi(wvbot,im,jm)
        call yperi3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
        call yperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)

        if (ipery.lt.0) then !free-slip east&west
           if(n_east.eq.-1)           then 
              wvbot(im,:)=wvbot(imm1,:)
              do k=1,kbm1
               vf(im,:,k)=vf(imm1,:,k)
              enddo
            endif
           if(n_west.eq.-1)           then
              wvbot(1,:)=wvbot(2,:)        
              do k=1,kbm1
               vf(1,:,k)=vf(2,:,k)
              enddo
            endif
         endif
        endif ! end if (ipery.ne.0)

         do k=1,kbm1
            do j=1,jm
               do i=1,im
                  uf(i,j,k)=uf(i,j,k)*dum(i,j)
                  vf(i,j,k)=vf(i,j,k)*dvm(i,j)
               end do
            end do
         end do

        return
*/
	else if (idx == 3){
//!     EAST
//!     radiation boundary conditions.
		if (n_east == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					ga = sqrtf(h[j][im-1]/hmax);
					uf[k][j][im-1] = ga*(0.25f*u[k][j-1][imm1-1]
										+0.5f*u[k][j][imm1-1]
										+0.25f*u[k][j+1][imm1-1])
									+(1.f-ga)*(0.25f*u[k][j-1][im-1]
											  +0.5f*u[k][j][im-1]
											  +0.25f*u[k][j+1][im-1]);
					vf[k][j][im-1] = 0;
				}
			}
		}

//!     WEST
//!     radiation boundary conditions.
		if (n_west == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 1; j < jmm1; j++){
					ga = sqrtf(h[j][0]/hmax);
					uf[k][j][1] = ga*(0.25f*u[k][j-1][2]
									 +0.5f*u[k][j][2]
									 +0.25f*u[k][j+1][2])
								 +(1.f-ga)*(0.25f*u[k][j-1][1]
										   +0.5f*u[k][j][1]
										   +0.25f*u[k][j+1][1]);
					uf[k][j][0] = uf[k][j][1];
					vf[k][j][0] = 0;
				}
			}
		}

//!     NORTH
//!     radiation boundary conditions.
		
		if (n_north == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 1; i < imm1; i++){
					ga = sqrtf(h[jm-1][i]/hmax);
					vf[k][jm-1][i] = ga*(0.25f*v[k][jmm1-1][i-1]
										+0.5f*v[k][jmm1-1][i]
										+0.25f*v[k][jmm1-1][i+1])
									+(1.f-ga)*(0.25f*v[k][jm-1][i-1]
											  +0.5f*v[k][jm-1][i]
											  +0.25f*v[k][jm-1][i+1]);
					uf[k][jm-1][i] = 0;
				}
			}
		}

//!     SOUTH
//!     radiation boundary conditions.
		
		if (n_south == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 1; i < imm1; i++){
					ga = sqrtf(h[0][i]/hmax);	
					vf[k][1][i] = ga*(0.25f*v[k][2][i-1]
									 +0.5f*v[k][2][i]
									 +0.25*v[k][2][i+1])
								 +(1.f-ga)*(0.25f*v[k][1][i-1]
										   +0.5f*v[k][1][i]
										   +0.25f*v[k][1][i+1]);
					vf[k][0][i] = vf[k][1][i];
					uf[k][0][i] = 0;
				}
			}
		}

		if (iperx != 0){
			xperi2d_mpi(wubot, im, jm);
			xperi2d_mpi(wvbot, im, jm);
			xperi3d_mpi(uf, im, jm, kbm1);
			xperi3d_mpi(vf, im, jm, kbm1);

			if (iperx < 0){
				if (n_north == -1){
					for (i = 0; i < im; i++){
						wubot[jm-1][i] = wubot[jmm1-1][i];	
					}
					for (k = 0; k < kbm1; k++){
						for (i = 0; i < im; i++){
							uf[k][jm-1][i] = uf[k][jmm1-1][i];	
						}
					}
				}

				if (n_south == -1){
					for (i = 0; i < im; i++){
						wubot[0][i] = wubot[1][i];	
					}
					for (k = 0; k < kbm1; k++){
						for (i = 0; i < im; i++){
							uf[k][0][i] = uf[k][1][i];	
						}
					}
				}
				
			}

		}

		if (ipery != 0){
			yperi2d_mpi(wubot, im, jm);
			yperi2d_mpi(wvbot, im, jm);
			yperi3d_mpi(uf, im, jm, kbm1);
			yperi3d_mpi(vf, im, jm, kbm1);

			if (ipery < 0){
				if (n_east == -1){
					for (j = 0; j < jm; j++){
						wvbot[j][im-1] = wvbot[j][imm1-1];	
					}
					for (k = 0; k < kbm1; k++){
						for (j = 0; j < jm; j++){
							vf[k][j][im-1] = vf[k][j][imm1-1];	
						}
					}
				}

				if (n_west == -1){
					for (j = 0; j < jm; j++){
						wvbot[j][0] = wvbot[j][1];	
					}
					for (k = 0; k < kbm1; k++){
						for (j = 0; j < jm; j++){
							vf[k][j][0] = vf[k][j][1];	
						}
					}
				}
			}
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*dum[j][i];	
					vf[k][j][i] = vf[k][j][i]*dvm[j][i];	
				}
			}
		}
	}



/*
      else if(idx.eq.4) then

! temperature and salinity boundary conditions (using uf and vf,
! respectively)

!    west 
      if(n_west.eq.-1) then

      do k=1,kbm1
      do j=1,jm 
      u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
      if(u1.ge.0.e0) then
      uf(1,j,k)=t(1,j,k)-u1*(t(1,j,k)-tbw(j,k))
      vf(1,j,k)=s(1,j,k)-u1*(s(1,j,k)-sbw(j,k))
      else
      uf(1,j,k)=t(1,j,k)-u1*(t(2,j,k)-t(1,j,k))
      vf(1,j,k)=s(1,j,k)-u1*(s(2,j,k)-s(1,j,k))
      if(k.ne.1.and.k.ne.kbm1) then
      wm=.5e0*(w(2,j,k)+w(2,j,k+1))*dti
     $   /((zz(k-1)-zz(k+1))*dt(2,j))
      uf(1,j,k)=uf(1,j,k)-wm*(t(2,j,k-1)-t(2,j,k+1))
      vf(1,j,k)=vf(1,j,k)-wm*(s(2,j,k-1)-s(2,j,k+1))
      end if
      end if !endif u1
      enddo  !enddo j
      enddo  !enddo k 

      if(nfw.gt.3) then  !west FRZ needs at least 4 pts
      do k=1,kbm1; do j=1,jm; do i=1,nfw
      uf(i,j,k)=uf(i,j,k)*(1.-frz(i,j))+(tobw(i,j,k)*frz(i,j))
      vf(i,j,k)=vf(i,j,k)*(1.-frz(i,j))+(sobw(i,j,k)*frz(i,j))
      enddo;enddo;enddo
      end if !if(nfw.gt.3) then..

      end if !endif west
!
!     east
      if(n_east.eq.-1) then

      do k=1,kbm1
      do j=1,jm
      u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
      if(u1.le.0.e0) then
      uf(im,j,k)=t(im,j,k)-u1*(tbe(j,k)-t(im,j,k))
      vf(im,j,k)=s(im,j,k)-u1*(sbe(j,k)-s(im,j,k))
      else
      uf(im,j,k)=t(im,j,k)-u1*(t(im,j,k)-t(imm1,j,k))
      vf(im,j,k)=s(im,j,k)-u1*(s(im,j,k)-s(imm1,j,k))
      if(k.ne.1.and.k.ne.kbm1) then
      wm=.5e0*(w(imm1,j,k)+w(imm1,j,k+1))*dti
     $    /((zz(k-1)-zz(k+1))*dt(imm1,j))
      uf(im,j,k)=uf(im,j,k)-wm*(t(imm1,j,k-1)-t(imm1,j,k+1))
      vf(im,j,k)=vf(im,j,k)-wm*(s(imm1,j,k-1)-s(imm1,j,k+1))
      end if
      end if !endif u1
      enddo  !enddo j
      enddo  !enddo k

      if(nfe.gt.3) then  !east FRZ needs at least 4 pts
      do k=1,kbm1; do j=1,jm; do i=1,nfe
      ii=im-i+1
      uf(ii,j,k)=uf(ii,j,k)*(1.-frz(ii,j))+(tobe(i,j,k)*frz(ii,j))
      vf(ii,j,k)=vf(ii,j,k)*(1.-frz(ii,j))+(sobe(i,j,k)*frz(ii,j))
      enddo;enddo;enddo
      end if !if(nfe.gt.3) then..

      end if !endif east
!
!   north
      if(n_north.eq.-1) then

      do k=1,kbm1
      do i=1,im
      u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
      if(u1.le.0.e0) then
      uf(i,jm,k)=t(i,jm,k)-u1*(tbn(i,k)-t(i,jm,k))
      vf(i,jm,k)=s(i,jm,k)-u1*(sbn(i,k)-s(i,jm,k))
      else
      uf(i,jm,k)=t(i,jm,k)-u1*(t(i,jm,k)-t(i,jmm1,k))
      vf(i,jm,k)=s(i,jm,k)-u1*(s(i,jm,k)-s(i,jmm1,k))
      if(k.ne.1.and.k.ne.kbm1) then
      wm=.5e0*(w(i,jmm1,k)+w(i,jmm1,k+1))*dti
     $   /((zz(k-1)-zz(k+1))*dt(i,jmm1))
      uf(i,jm,k)=uf(i,jm,k)-wm*(t(i,jmm1,k-1)-t(i,jmm1,k+1))
      vf(i,jm,k)=vf(i,jm,k)-wm*(s(i,jmm1,k-1)-s(i,jmm1,k+1))
      end if
      end if !endif u1
      end do !enddo i
      end do !enddo k

      if(nfn.gt.3) then  !east FRZ needs at least 4 pts
      do k=1,kbm1; do i=1,im; do j=1,nfn
      jj=jm-j+1
      uf(i,jj,k)=uf(i,jj,k)*(1.-frz(i,jj))+(tobn(i,j,k)*frz(i,jj))
      vf(i,jj,k)=vf(i,jj,k)*(1.-frz(i,jj))+(sobn(i,j,k)*frz(i,jj))
      enddo;enddo;enddo
      end if !if(nfn.gt.3) then..

      end if !endif north
!
!    south
      if(n_south.eq.-1) then

      do k=1,kbm1
      do i=1,im 
      u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
      if(u1.ge.0.e0) then
      uf(i,1,k)=t(i,1,k)-u1*(t(i,1,k)-tbs(i,k))
      vf(i,1,k)=s(i,1,k)-u1*(s(i,1,k)-sbs(i,k))
      else
      uf(i,1,k)=t(i,1,k)-u1*(t(i,2,k)-t(i,1,k))
      vf(i,1,k)=s(i,1,k)-u1*(s(i,2,k)-s(i,1,k))
      if(k.ne.1.and.k.ne.kbm1) then
      wm=.5e0*(w(i,2,k)+w(i,2,k+1))*dti
     $   /((zz(k-1)-zz(k+1))*dt(i,2))
      uf(i,1,k)=uf(i,1,k)-wm*(t(i,2,k-1)-t(i,2,k+1))
      vf(i,1,k)=vf(i,1,k)-wm*(s(i,2,k-1)-s(i,2,k+1))
      end if
      end if !endif u1
      enddo  !enddo i
      enddo  !enddo k

      if(nfs.gt.3) then  !east FRZ needs at least 4 pts
      do k=1,kbm1; do i=1,im; do j=1,nfs
      uf(i,j,k)=(uf(i,j,k)*(1.-frz(i,j)))+(tobs(i,j,k)*frz(i,j))
      vf(i,j,k)=(vf(i,j,k)*(1.-frz(i,j)))+(sobs(i,j,k)*frz(i,j))
      enddo;enddo;enddo
      end if !if(nfs.gt.3) then..

      end if !endif south
!
        if (iperx.ne.0) then
          call xperi3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
          call xperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)
        endif
!
        if (ipery.ne.0) then
          call yperi3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
          call yperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)
        endif
!         
         do k=1,kbm1
            do j=1,jm
               do i=1,im
                  uf(i,j,k)=uf(i,j,k)*fsm(i,j)
                  vf(i,j,k)=vf(i,j,k)*fsm(i,j)
               end do
            end do
         end do
        
        return
*/
	else if (idx == 4){

		if (n_west == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][1]*dti/(dx[j][0]+dx[j][1]);
					if (u1 >= 0.0f){
						uf[k][j][0] = t[k][j][0]-u1*(t[k][j][0]-tbw[k][j]);	
						vf[k][j][0] = s[k][j][0]-u1*(s[k][j][0]-sbw[k][j]);
					}else{
						uf[k][j][0] = t[k][j][0]-u1*(t[k][j][1]-t[k][j][0]);	
						vf[k][j][0] = s[k][j][0]-u1*(s[k][j][1]-s[k][j][0]);
						if (k != 0 && k != kbm1-1){
							wm = 0.5f*(w[k][j][1]+w[k+1][j][1])*dti/
									((zz[k-1]-zz[k+1])*dt[j][1]);
							uf[k][j][0] = uf[k][j][0]-wm*(t[k-1][j][1]-t[k+1][j][1]);
							vf[k][j][0] = vf[k][j][0]-wm*(s[k-1][j][1]-s[k+1][j][1]);
						}
					}
				}
			}

			if (nfw > 3){
				for (k = 0; k < kbm1; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < nfw; i++){
							uf[k][j][i] = uf[k][j][i]*(1.f-frz[j][i])
										 +(tobw[k][j][i]*frz[j][i]);
							vf[k][j][i] = vf[k][j][i]*(1.f-frz[j][i])
										 +(sobw[k][j][i]*frz[j][i]);
						}
					}
				}
			}
		}

//!     east
		if (n_east == -1){
			for (k = 0; k < kbm1; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][im-1]*dti/(dx[j][im-1]+dx[j][imm1-1]);
					if (u1 <= 0.0f){
						uf[k][j][im-1] = t[k][j][im-1]-u1*(tbe[k][j]-t[k][j][im-1]);	
						vf[k][j][im-1] = s[k][j][im-1]-u1*(sbe[k][j]-s[k][j][im-1]);
					}else{
						uf[k][j][im-1] = t[k][j][im-1]-u1*(t[k][j][im-1]-t[k][j][imm1-1]);	
						vf[k][j][im-1] = s[k][j][im-1]-u1*(s[k][j][im-1]-s[k][j][imm1-1]);
						if (k != 0 && k != kbm1-1){
							wm = 0.5f*(w[k][j][imm1-1]+w[k+1][j][imm1-1])*dti/
								   ((zz[k-1]-zz[k+1])*dt[j][imm1-1]);
							uf[k][j][im-1] = uf[k][j][im-1]-wm*(t[k-1][j][imm1-1]-t[k+1][j][imm1-1]);
							vf[k][j][im-1] = vf[k][j][im-1]-wm*(s[k-1][j][imm1-1]-s[k+1][j][imm1-1]);
						}
					}
				}
			}

			if (nfe > 3){
				for (k = 0; k < kbm1; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < nfe; i++){
							int ii = im-i-1;	
							uf[k][j][ii] = uf[k][j][ii]
											*(1.f-frz[j][ii])
										  +(tobe[k][j][i]*frz[j][ii]);
							vf[k][j][ii] = vf[k][j][ii]
											*(1.f-frz[j][ii])
										  +(sobe[k][j][i]*frz[j][ii]);
						}
					}
				}
			}
		}

//north
		if (n_north == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][jm-1][i]*dti/(dy[jm-1][i]+dy[jmm1-1][i]);	
					if (u1 <= 0.0f){
						uf[k][jm-1][i] = t[k][jm-1][i]-u1*(tbn[k][i]-t[k][jm-1][i]);
						vf[k][jm-1][i] = s[k][jm-1][i]-u1*(sbn[k][i]-s[k][jm-1][i]);
					}else{
						uf[k][jm-1][i] = t[k][jm-1][i]-u1*(t[k][jm-1][i]-t[k][jmm1-1][i]);
						vf[k][jm-1][i] = s[k][jm-1][i]-u1*(s[k][jm-1][i]-s[k][jmm1-1][i]);
						if (k != 0 && k != kbm1-1){
							wm = 0.5f*(w[k][jmm1-1][i]+w[k+1][jmm1-1][i])*dti/
									((zz[k-1]-zz[k+1])*dt[jmm1-1][i]);	
							uf[k][jm-1][i] = uf[k][jm-1][i]-wm*(t[k-1][jmm1-1][i]-t[k+1][jmm1-1][i]);
							vf[k][jm-1][i] = vf[k][jm-1][i]-wm*(s[k-1][jmm1-1][i]-s[k+1][jmm1-1][i]);
						}
					}
				}
			}

			if (nfn > 3){
				for (k = 0; k < kbm1; k++){
					for (i = 0; i < im; i++){
						for (j = 0; j < nfn; j++){
							int jj = jm-j-1;	
							uf[k][jj][i] = uf[k][jj][i]
											*(1.f-frz[jj][i])
										  +(tobn[k][j][i]*frz[jj][i]);
							vf[k][jj][i] = vf[k][jj][i]
											*(1.f-frz[jj][i])
										  +(sobn[k][j][i]*frz[jj][i]);
						}
					}
				}
			}
		}

//!    south
		if (n_south == -1){
			for (k = 0; k < kbm1; k++){
				for (i = 0; i < im; i++){
					u1=2.0f*v[k][1][i]*dti/(dy[0][i]+dy[1][i]);	
					if (u1 >= 0.0f){
						uf[k][0][i] = t[k][0][i]-u1*(t[k][0][i]-tbs[k][i]);	
						vf[k][0][i] = s[k][0][i]-u1*(s[k][0][i]-sbs[k][i]);
					}else{
						uf[k][0][i] = t[k][0][i]-u1*(t[k][1][i]-t[k][0][i]);	
						vf[k][0][i] = s[k][0][i]-u1*(s[k][1][i]-s[k][0][i]);
						if (k != 0 && k != kbm1-1){
							wm = 0.5f*(w[k][1][i]+w[k+1][1][i])*dti/
									((zz[k-1]-zz[k+1])*dt[1][i]);
							uf[k][0][i] = uf[k][0][i]-wm*(t[k-1][1][i]-t[k+1][1][i]);
							vf[k][0][i] = vf[k][0][i]-wm*(s[k-1][1][i]-s[k+1][1][i]);
						}
					}
				}
			}

			if (nfs > 3){
				for (k = 0; k < kbm1; k++){
					for (i = 0; i < im; i++){
						for (j = 0; j < nfs; j++){
							uf[k][j][i] = (uf[k][j][i]*(1.f-frz[j][i]))	
										 +(tobs[k][j][i]*frz[j][i]);
							vf[k][j][i] = (vf[k][j][i]*(1.f-frz[j][i]))
										 +(sobs[k][j][i]*frz[j][i]);
						}
					}
				}
			}
		}

		if (iperx != 0){
			xperi3d_mpi(uf, im, jm, kbm1);	
			xperi3d_mpi(vf, im, jm, kbm1);	
		}

		if (ipery != 0){
			yperi3d_mpi(uf, im, jm, kbm1);	
			yperi3d_mpi(vf, im, jm, kbm1);	
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*fsm[j][i];
					vf[k][j][i] = vf[k][j][i]*fsm[j][i];
				}
			}
		}

	}

/*
      else if(idx.eq.5) then

! vertical velocity boundary conditions

        if (iperx.ne.0) then
         call xperi3d_mpi(w(:,:,1:kbm1),im,jm,kbm1)
        endif

        if (ipery.ne.0) then
         call yperi3d_mpi(w(:,:,1:kbm1),im,jm,kbm1)
        endif

        do k=1,kbm1
          do j=1,jm
            do i=1,im
              w(i,j,k)=w(i,j,k)*fsm(i,j)
            end do
          end do
        end do

        return
*/
	else if (idx == 5){
		if (iperx != 0){
			xperi3d_mpi(w, im, jm, kbm1);
		}

		if (ipery != 0){
			yperi3d_mpi(w, im, jm, kbm1);
		}

		for (k = 0; k < kbm1; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					w[k][j][i] *= fsm[j][i];	
				}
			}
		}
	}

/*
      else if(idx.eq.6) then

! q2 and q2l boundary conditions

! east
       if(n_east.eq.-1) then

         do k=1,kb
           do j=1,jm
              u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
              if(u1.le.0.e0) then
                uf(im,j,k)=q2(im,j,k)-u1*(small-q2(im,j,k))
                vf(im,j,k)=q2l(im,j,k)-u1*(small-q2l(im,j,k))
              else
                uf(im,j,k)=q2(im,j,k)-u1*(q2(im,j,k)-q2(imm1,j,k))
                vf(im,j,k)=q2l(im,j,k)-u1*(q2l(im,j,k)-q2l(imm1,j,k))
              endif
           enddo
         enddo

       end if
                       
! west
       if(n_west.eq.-1) then

         do k=1,kb
           do j=1,jm
              u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
              if(u1.ge.0.e0) then
                uf(1,j,k)=q2(1,j,k)-u1*(q2(1,j,k)-small)
                vf(1,j,k)=q2l(1,j,k)-u1*(q2l(1,j,k)-small)
              else
                uf(1,j,k)=q2(1,j,k)-u1*(q2(2,j,k)-q2(1,j,k))
                vf(1,j,k)=q2l(1,j,k)-u1*(q2l(2,j,k)-q2l(1,j,k))
              endif
           enddo
         enddo 


       end if

! north
       if(n_north.eq.-1) then

         do k=1,kb
           do i=1,im
              u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
              if(u1.le.0.e0) then
                uf(i,jm,k)=q2(i,jm,k)-u1*(small-q2(i,jm,k))
                vf(i,jm,k)=q2l(i,jm,k)-u1*(small-q2l(i,jm,k))
              else
                uf(i,jm,k)=q2(i,jm,k)-u1*(q2(i,jm,k)-q2(i,jmm1,k))
                vf(i,jm,k)=q2l(i,jm,k)-u1*(q2l(i,jm,k)-q2l(i,jmm1,k))
              endif
           enddo
         end do

       endif


! south
       if(n_south.eq.-1) then

         do k=1,kb
           do i=1,im
              u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
              if(u1.ge.0.e0) then
                uf(i,1,k)=q2(i,1,k)-u1*(q2(i,1,k)-small)
                vf(i,1,k)=q2l(i,1,k)-u1*(q2l(i,1,k)-small)
              else
                uf(i,1,k)=q2(i,1,k)-u1*(q2(i,2,k)-q2(i,1,k))
                vf(i,1,k)=q2l(i,1,k)-u1*(q2l(i,2,k)-q2l(i,1,k))
              endif
           enddo
         enddo


       end if
!
        if (iperx.ne.0) then
         call xperi3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
         call xperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)
         call xperi3d_mpi(kh(:,:,1:kbm1),im,jm,kbm1)
         call xperi3d_mpi(km(:,:,1:kbm1),im,jm,kbm1)
         call xperi3d_mpi(kq(:,:,1:kbm1),im,jm,kbm1)
         call xperi3d_mpi(l(:,:,1:kbm1),im,jm,kbm1)
        endif

        if (ipery.ne.0) then
         call yperi3d_mpi(uf(:,:,1:kbm1),im,jm,kbm1)
         call yperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)
         call yperi3d_mpi(kh(:,:,1:kbm1),im,jm,kbm1)
         call yperi3d_mpi(km(:,:,1:kbm1),im,jm,kbm1)
         call yperi3d_mpi(kq(:,:,1:kbm1),im,jm,kbm1)
         call yperi3d_mpi(l(:,:,1:kbm1),im,jm,kbm1)
        endif

        do k=1,kb
          do j=1,jm
            do i=1,im
              uf(i,j,k)=uf(i,j,k)*fsm(i,j)
              vf(i,j,k)=vf(i,j,k)*fsm(i,j)
            end do
          end do
        end do

        return
*/
	else if (idx == 6){
		if (n_east == -1){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][im-1]*dti
						/(dx[j][im-1]+dx[j][imm1-1]);	
					if (u1 <= 0.0f){
						uf[k][j][im-1] = q2[k][j][im-1]-u1*(small-q2[k][j][im-1]);
						vf[k][j][im-1] = q2l[k][j][im-1]-u1*(small-q2l[k][j][im-1]);
					}else{
						uf[k][j][im-1] = q2[k][j][im-1]
							         -u1*(q2[k][j][im-1]-q2[k][j][imm1-1]);	

						vf[k][j][im-1] = q2l[k][j][im-1]
							         -u1*(q2l[k][j][im-1]-q2l[k][j][imm1-1]);
					}
				}
			}
		}

		if (n_west == -1){
			for (k = 0; k < kb; k++){
				for (j = 0; j < jm; j++){
					u1 = 2.0f*u[k][j][1]*dti/(dx[j][0]+dx[j][1]);	
					if (u1 >= 0.0f){
						uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][0]-small);
						vf[k][j][0] = q2l[k][j][0]-u1*(q2l[k][j][0]-small);
					}else{
						uf[k][j][0] = q2[k][j][0]-u1*(q2[k][j][1]-q2[k][j][0]);	
						vf[k][j][0] = q2l[k][j][0] - u1*(q2l[k][j][1]-q2l[k][j][0]);
					}
				}
			}
		}

		if (n_north == -1){
			for (k = 0; k < kb; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][jm-1][i]*dti/(dy[jm-1][i]+dy[jmm1-1][i]);		
					if (u1 <= 0.0f){
						uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(small-q2[k][jm-1][i]);	
						vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(small-q2l[k][jm-1][i]);
					}else{
						uf[k][jm-1][i] = q2[k][jm-1][i]-u1*(q2[k][jm-1][i]-q2[k][jmm1-1][i]);	
						vf[k][jm-1][i] = q2l[k][jm-1][i]-u1*(q2l[k][jm-1][i]-q2l[k][jmm1-1][i]);
					}

				}
			}
		}

		if (n_south == -1){
			for (k = 0; k < kb; k++){
				for (i = 0; i < im; i++){
					u1 = 2.0f*v[k][1][i]*dti/(dy[0][i]+dy[1][i]);	
					if (u1 >= 0.0f){
						uf[k][0][i] = q2[k][0][i]-u1*(q2[k][0][i]-small);	
						vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][0][i]-small);
					}else{
						uf[k][0][i] = q2[k][0][i]-u1*(q2[k][1][i]-q2[k][0][i]);	
						vf[k][0][i] = q2l[k][0][i]-u1*(q2l[k][1][i]-q2l[k][0][i]);
					}
				}
			}
		}
		
		if (iperx != 0){
			xperi3d_mpi(uf, im, jm, kbm1);	
			xperi3d_mpi(vf, im, jm, kbm1);	
			xperi3d_mpi(kh, im, jm, kbm1);	
			xperi3d_mpi(km, im, jm, kbm1);	
			xperi3d_mpi(kq, im, jm, kbm1);	
			xperi3d_mpi(l, im, jm, kbm1);	
		}

		if (ipery != 0){
			yperi3d_mpi(uf, im, jm, kbm1);
			yperi3d_mpi(vf, im, jm, kbm1);
			yperi3d_mpi(kh, im, jm, kbm1);
			yperi3d_mpi(km, im, jm, kbm1);
			yperi3d_mpi(kq, im, jm, kbm1);
			yperi3d_mpi(l, im, jm, kbm1);
		}

		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					uf[k][j][i] = uf[k][j][i]*fsm[j][i];
					vf[k][j][i] = vf[k][j][i]*fsm[j][i];
				}
			}
		}

	}

/*
       else if(idx.eq.7) then

!    east

        if(n_east.eq.-1) then

            do k=1,kbm1
               do j=1,jm
                  u1=2.e0*u(im,j,k)*dti/(dx(im,j)+dx(imm1,j))
                  if(u1.le.0.e0) then
                     vf(im,j,k)=
     $               tr(im,j,k,inb)-u1*(tre(j,k,inb)-tr(im,j,k,inb))
                  else
                     vf(im,j,k)=
     $               tr(im,j,k,inb)-u1*(tr(im,j,k,inb)-tr(imm1,j,k,inb))
                  end if
               enddo
            enddo

        endif

!    west

        if(n_west.eq.-1) then

            do k=1,kbm1
               do j=1,jm
                  u1=2.e0*u(2,j,k)*dti/(dx(1,j)+dx(2,j))
                  if(u1.ge.0.e0) then
                     vf(1,j,k)=
     $               tr(1,j,k,inb)-u1*(tr(1,j,k,inb)-trw(j,k,inb))
                  else
                     vf(1,j,k)=
     $               tr(1,j,k,inb)-u1*(tr(2,j,k,inb)-tr(1,j,k,inb))
                  end if
               enddo
            enddo

        endif

!    north

        if(n_north.eq.-1) then

            do k=1,kbm1
               do i=1,im
                  u1=2.e0*v(i,jm,k)*dti/(dy(i,jm)+dy(i,jmm1))
                  if(u1.le.0.e0) then
                     vf(i,jm,k)=
     $               tr(i,jm,k,inb)-u1*(trn(i,k,inb)-tr(i,jm,k,inb))
                  else
                     vf(i,jm,k)=
     $               tr(i,jm,k,inb)-u1*(tr(i,jm,k,inb)-tr(i,jmm1,k,inb))
                  end if
               enddo
            enddo

        endif

!    south

        if(n_south.eq.-1) then

            do k=1,kbm1
               do i=1,im
                  u1=2.e0*v(i,2,k)*dti/(dy(i,1)+dy(i,2))
                  if(u1.ge.0.e0) then
                     vf(i,1,k)=
     $               tr(i,1,k,inb)-u1*(tr(i,1,k,inb)-trs(i,k,inb))
                  else
                     vf(i,1,k)=
     $               tr(i,1,k,inb)-u1*(tr(i,2,k,inb)-tr(i,1,k,inb))
                  end if
               enddo
            enddo

        endif
!
        if (iperx.ne.0) then
          call xperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)
        endif
!
        if (ipery.ne.0) then
          call yperi3d_mpi(vf(:,:,1:kbm1),im,jm,kbm1)
        endif
!         
         do k=1,kbm1
            do j=1,jm
               do i=1,im                  
                  vf(i,j,k)=vf(i,j,k)*fsm(i,j)
               end do
            end do
         end do

        return

      endif

      end
*/
}

