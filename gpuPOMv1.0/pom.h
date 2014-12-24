! pom.h

! contain parameters for the model domain and for the decomposition
! local domain size, and common POM variables

!_______________________________________________________________________
! Grid size parameters
      integer
     $  im_global      ,! number of global grid points in x
     $  jm_global      ,! number of global grid points in y
     $  kb             ,! number of grid points in z
     $  im_local       ,! number of local grid points in x
     $  jm_local       ,! number of local grid points in y
     $  im_global_coarse,! number of global grid points in x for coarse grids
     $  jm_global_coarse,! number of global grid points in y for coarse grids
     $  im_local_coarse ,
     $  jm_local_coarse ,
     $  x_division      ,! number of divisions from coarse to fine grids in x 
     $  y_division      ,! number of divisions from coarse to fine grids in y 
     $  n_proc           ! number of processors
     

! Correct values for im_local and jm_local are found using
!   n_proc=(im_global-2)/(im_local-2)*(jm_global-2)/(jm_local-2)
! Values higher than necessary will not cause the code to fail, but
! will allocate more memory than is necessary. Value that are too low
! will cause the code to exit
! x_divison and y_division can change according to the nest requirement
!  must >=2 for otherwise "if" statements in initialize.f won't work

      parameter(
     $  im_global=202        ,
     $  jm_global= 47        ,
     $  kb=3                 ,
     $  im_local=52          ,
     $  jm_local=47          ,
     $  im_global_coarse=202 ,
     $  jm_global_coarse= 47 ,
     $  im_local_coarse=52   ,
     $  jm_local_coarse=47   ,
     $  x_division=2         ,     
     $  y_division=2         ,    
     $  n_proc=4              )


! tide parameters !fhx:tide
      integer
     $  ntide            ! number of tidal components  
      parameter(ntide=1) ! =0 may not work below & in solver.f !lyo:pac10:

! Efective grid size
      integer
     $  im             ,! number of grid points used in local x domains
     $  imm1           ,! im-1
     $  imm2           ,! im-2
     $  jm             ,! number of grid points used in local y domains
     $  jmm1           ,! jm-1
     $  jmm2           ,! jm-2
     $  kbm1           ,! kb-1
     $  kbm2           ,! kb-2
     $  im_coarse      ,! number of coarse grid points used in local x domains
     $  jm_coarse       ! number of coarse grid points used in local y domains

! Note that im and jm may be different between local domains
! im and jm are equal to or lower than im_local and jm_local, depending
! on the use of correct values for im_local and jm_local

      common/blksiz/
     $  im             ,
     $  imm1           ,
     $  imm2           ,
     $  jm             ,
     $  jmm1           ,
     $  jmm2           ,
     $  kbm1           ,
     $  kbm2           ,
     $  im_coarse       ,
     $  jm_coarse

!_______________________________________________________________________
! Parallel variables
      integer
     $  my_task        ,! actual parallel processor ID
     $  master_task    ,! master processor ID
     $  pom_comm       ,! POM model MPI group communicator
     $  i_global       ,! global i index for each point in local domain
     $  j_global       ,! global j index for each point in local domain
     $  pom_comm_coarse ,! satellite data MPI group communicator
     $  i_global_coarse ,! global i index for each point in local domain
     $  j_global_coarse ,! global j index for each point in local domain     
     $  n_west         ,! western parallel processor ID
     $  n_east         ,! eastern parallel processor ID
     $  n_south        ,! southern parallel processor ID
     $  n_north         ! northern parallel processor ID

      common/blkpar/
     $  my_task        ,
     $  master_task    ,
     $  pom_comm       ,
     $  i_global(im_local),
     $  j_global(jm_local),
     $  pom_comm_coarse ,
     $  i_global_coarse (im_local_coarse),
     $  j_global_coarse (jm_local_coarse),
     $  n_west         ,
     $  n_east         ,
     $  n_south        ,
     $  n_north

!_______________________________________________________________________
! Scalars
      integer
     $  iint           ,
     $  iprint         ,! interval in iint at which variables are printed
     $  mode           ,! calculation mode
     $  ntp            ,! water type
     $  iend           ,! total internal mode time steps
     $  iext           ,
     $  ispadv         ,! step interval for updating external advective terms
     $  isplit         ,! dti/dte
     $  nadv           ,! advection scheme
     $  nbct           ,! surface temperature boundary condition
     $  nbcs           ,! surface salinity boundary condition
     $  nitera         ,! number of iterations for Smolarkiewicz scheme
     $  npg            ,! pressure gradient scheme !fhx:Toni:npg
     $  nread_rst      ,! index to start from restart file
     $  irestart       ,
     $  iperx          ,!periodic boundary condition in x direction
     $  ipery          ,!periodic boundary condition in y direction
     $  n1d            ,!n1d .ne. 0 for 1d-simulation
     $  ngrid          ,!ngrid .gt. 0 use grid.nc,<=0 to call specify_grid
     $  nse            ,! sponge pnts next to bound e
     $  nsw            ,! sponge pnts next to bound w
     $  nsn            ,! sponge pnts next to bound n
     $  nss            ,! sponge pnts next to bound s
     $  error_status

      real
     $  alpha          ,! weight for surface slope term in external eq
     $  dte            ,! external (2-D) time step (s)
     $  dti            ,! internal (3-D) time step (s)
     $  dti2           ,! 2*dti
     $  grav           ,! gravity constant (S.I. units)
     $  kappa          ,! von Karman's constant
     $  pi             ,! pi
     $  deg2rad        ,! pi/180.
     $  ramp           ,! inertial ramp
     $  rfe            ,! flag for eastern open boundary (see bcond)
     $  rfn            ,! flag for northern open boundary (see bcond)
     $  rfs            ,! flag for southern open boundary (see bcond)
     $  rfw            ,! flag for western open boundary (see bcond)
     $  rhoref         ,! reference density
     $  sbias          ,! salinity bias
     $  slmax          ,
     $  small          ,! small value
     $  tbias          ,! temperature bias
     $  time           ,! model time (days)
     $  tprni          ,! inverse horizontal turbulent Prandtl number
     $  umol           ,! background viscosity
     $  vmaxl          ,! max vaf used to test for model blow-up
     $  write_rst      ,
     $  aam_init       ,! initial value of aam
     $  cbcmax         ,! maximum bottom friction coefficient
     $  cbcmin         ,! minimum bottom friction coefficient
     $  days           ,! run duration in days
     $  dte2           ,! 2*dte
     $  horcon         ,! smagorinsky diffusivity coefficient
     $  ispi           ,! dte/dti
     $  isp2i          ,! dte/(2*dti)
     $  prtd1          ,! initial print interval (days)
     $  prtd2          ,! final print interval (days)
     $  smoth          ,! constant to prevent solution splitting
     $  sw             ,! smoothing parameter for Smolarkiewicz scheme
     $  time0          ,! initial time (days)
     $  z0b            ,! bottom roughness
     $  lono,lato      ,! lon,lat where aam*(1+fak) larger influence (xs,ys)
     $  xs,ys,fak      ,! set lono or lato=999.     to skip 
     $  alonc,alatc     ! Center lon/lat @ (im_global/2,jm_global/2)
!    $  period         ,! inertial period

      parameter(lono=999.0,lato=999.0,xs=1.5,ys=1.5,fak=0.5)
!     parameter(lono=141.2,lato= 41.8,xs=1.5,ys=1.5,fak=0.4)
!     parameter(lono=141.2,lato= 41.8,xs=1.5,ys=1.5,fak=1.0)
			  !     (i,j)=(427, 575)

      common/blkcon/
     $  alpha          ,
     $  dte            ,
     $  dti            ,
     $  dti2           ,
     $  grav           ,
     $  kappa          ,
     $  pi             ,
     $  deg2rad        ,
     $  ramp           ,
     $  rfe            ,
     $  rfn            ,
     $  rfs            ,
     $  rfw            ,
     $  rhoref         ,
     $  sbias          ,
     $  slmax          ,
     $  small          ,
     $  tbias          ,
     $  time           ,
     $  tprni          ,
     $  umol           ,
     $  vmaxl          ,
     $  write_rst      ,
     $  iint           ,
     $  iprint         ,
     $  mode           ,
     $  ntp            ,
     $  aam_init       ,
     $  cbcmax         ,
     $  cbcmin         ,
     $  days           ,
     $  dte2           ,
     $  horcon         ,
     $  ispi           ,
     $  isp2i          ,
     $  prtd1          ,
     $  prtd2          ,
     $  smoth          ,
     $  sw             ,
     $  time0          ,
     $  z0b            ,
     $  alonc          ,
     $  alatc          ,
     $  iend           ,
     $  iext           ,
     $  ispadv         ,
     $  isplit         ,
     $  nadv           ,
     $  nbct           ,
     $  nbcs           ,
     $  nitera         ,
     $  npg            ,
     $  nread_rst      ,
     $  irestart       ,
     $  iperx          ,
     $  ipery          ,
     $  n1d            ,
     $  ngrid          ,
     $  nse            ,! sponge pnts next to bound e
     $  nsw            ,! sponge pnts next to bound w
     $  nsn            ,! sponge pnts next to bound n
     $  nss            ,! sponge pnts next to bound s
     $  error_status
!     $  period         ,

!_______________________________________________________________________
! 1-D arrays
      real
     $  dz             ,! z(k)-z(k+1)
     $  dzz            ,! zz(k)-zz(k+1)
     $  z              ,! sigma coordinate from z=0 (surface) to z=-1 (bottom)
     $  zz              ! sigma coordinate, intermediate between z

      common/blk1d/ 
     $  dz(kb)         ,
     $  dzz(kb)        ,
     $  z(kb)          ,
     $  zz(kb)

	  real
     $	f_tmp_in
	  common/xsztest/
     $  f_tmp_in(im_local,jm_local)
!_______________________________________________________________________
! 2-D arrays
      real
     $  aam2d          ,! vertical average of aam
     $  aamfac         ,! aam factor for subroutine incmix             !fhx:incmix
     $  advua          ,! sum of the 2nd, 3rd and 4th terms in eq (18)
     $  advva          ,! sum of the 2nd, 3rd and 4th terms in eq (19)
     $  adx2d          ,! vertical integral of advx
     $  ady2d          ,! vertical integral of advy
     $  art            ,! cell area centered on T grid points
     $  aru            ,! cell area centered on U grid points
     $  arv            ,! cell area centered on V grid points
     $  cbc            ,! bottom friction coefficient
     $  cor            ,! coriolis parameter
     $  d              ,! h+el
     $  drx2d          ,! vertical integral of drhox
     $  dry2d          ,! vertical integral of drhoy
     $  dt             ,! h+et
     $  dum            ,! mask for u velocity
     $  dvm            ,! mask for v velocity
     $  dx             ,! grid spacing in x
     $  dy             ,! grid spacing in y
     $  east_c         ,! horizontal coordinate of cell corner points in x
     $  east_e         ,! horizontal coordinate of elevation points in x
     $  east_u         ,! horizontal coordinate of U points in x
     $  east_v         ,! horizontal coordinate of V points in x
     $  e_atmos        ,! atmospheric pressure
     $  egb            ,! surface elevation use for pressure gradient at time n-1
     $  egf            ,! surface elevation use for pressure gradient at time n+1
     $  el             ,! surface elevation used in the external mode at time n
     $  elb            ,! surface elevation used in the external mode at time n-1
     $  elf            ,! surface elevation used in the external mode at time n+1
     $  et             ,! surface elevation used in the internal mode at time n
     $  etb            ,! surface elevation used in the internal mode at time n-1
     $  etf            ,! surface elevation used in the internal mode at time n+1
     $  fluxua         ,
     $  fluxva         ,
     $  fsm            ,! mask for scalar variables
     $  h              ,! bottom depth
     $  north_c        ,! horizontal coordinate of cell corner points in y
     $  north_e        ,! horizontal coordinate of elevation points in y
     $  north_u        ,! horizontal coordinate of U points in y
     $  north_v        ,! horizontal coordinate of V points in y
     $  psi            ,
     $  rot            ,! rotation angle
     $  ssurf          ,
     $  swrad          ,! short wave radiation incident on the ocean surface
     $  vfluxb         ,! volume flux through water column surface at time n-1
     $  tps            ,
     $  tsurf          ,
     $  ua             ,! vertical mean of u at time n
     $  vfluxf         ,! volume flux through water column surface at time n+1
     $  uab            ,! vertical mean of u at time n-1
     $  uaf            ,! vertical mean of u at time n+1
     $  utb            ,! ua time averaged over the interval dti at time n-1
     $  utf            ,! ua time averaged over the interval dti at time n+1
     $  va             ,! vertical mean of v at time n
     $  vab            ,! vertical mean of v at time n-1
     $  vaf            ,! vertical mean of v at time n+1
     $  vtb            ,! va time averaged over the interval dti at time n-1
     $  vtf            ,! va time averaged over the interval dti at time n+1
     $  wssurf         ,! <ws(0)> salinity flux at the surface
     $  wtsurf         ,! <wt(0)> temperature flux at the surface
     $  wubot          ,! x-momentum flux at the bottom
     $  wusurf         ,! <wu(0)> momentum flux at the surface
     $  wvbot          ,! y-momentum flux at the bottom
     $  wvsurf         ,! <wv(0)> momentum flux at the surface
     $  alon_coarse     ,! elevation points in x for wind and satellite data
     $  alat_coarse     ,! elevation points in y for wind and satellite data
     $  mask_coarse       ! mask for scalar variables for wind and satellite data

      common/blk2d/
     $  aam2d(im_local,jm_local)   ,
     $  aamfac(im_local,jm_local)  ,    !fhx:incmix
     $  advua(im_local,jm_local)   ,
     $  advva(im_local,jm_local)   ,
     $  adx2d(im_local,jm_local)   ,
     $  ady2d(im_local,jm_local)   ,
     $  art(im_local,jm_local)     ,
     $  aru(im_local,jm_local)     ,
     $  arv(im_local,jm_local)     ,
     $  cbc(im_local,jm_local)     ,
     $  cor(im_local,jm_local)     ,
     $  d(im_local,jm_local)       ,
     $  drx2d(im_local,jm_local)   ,
     $  dry2d(im_local,jm_local)   ,
     $  dt(im_local,jm_local)      ,
     $  dum(im_local,jm_local)     ,
     $  dvm(im_local,jm_local)     ,
     $  dx(im_local,jm_local)      ,
     $  dy(im_local,jm_local)      ,
     $  east_c(im_local,jm_local)  ,
     $  east_e(im_local,jm_local)  ,
     $  east_u(im_local,jm_local)  ,
     $  east_v(im_local,jm_local)  ,
     $  e_atmos(im_local,jm_local) ,
     $  egb(im_local,jm_local)     ,
     $  egf(im_local,jm_local)     ,
     $  el(im_local,jm_local)      ,
     $  elb(im_local,jm_local)     ,
     $  elf(im_local,jm_local)     ,
     $  et(im_local,jm_local)      ,
     $  etb(im_local,jm_local)     ,
     $  etf(im_local,jm_local)     ,
     $  fluxua(im_local,jm_local)  ,
     $  fluxva(im_local,jm_local)  ,
     $  fsm(im_local,jm_local)     ,
     $  h(im_local,jm_local)       ,
     $  north_c(im_local,jm_local) ,
     $  north_e(im_local,jm_local) ,
     $  north_u(im_local,jm_local) ,
     $  north_v(im_local,jm_local) ,
     $  psi(im_local,jm_local)     ,
     $  rot(im_local,jm_local)     ,
     $  ssurf(im_local,jm_local)   ,
     $  swrad(im_local,jm_local)   ,
     $  vfluxb(im_local,jm_local)  ,
     $  tps(im_local,jm_local)     ,
     $  tsurf(im_local,jm_local)   ,
     $  ua(im_local,jm_local)      ,
     $  vfluxf(im_local,jm_local)  ,
     $  uab(im_local,jm_local)     ,
     $  uaf(im_local,jm_local)     ,
     $  utb(im_local,jm_local)     ,
     $  utf(im_local,jm_local)     ,
     $  va(im_local,jm_local)      ,
     $  vab(im_local,jm_local)     ,
     $  vaf(im_local,jm_local)     ,
     $  vtb(im_local,jm_local)     ,
     $  vtf(im_local,jm_local)     ,
     $  wssurf(im_local,jm_local)  ,
     $  wtsurf(im_local,jm_local)  ,
     $  wubot(im_local,jm_local)   ,
     $  wusurf(im_local,jm_local)  ,
     $  wvbot(im_local,jm_local)   ,
     $  wvsurf(im_local,jm_local)  ,
     $  alon_coarse(im_local_coarse,jm_local_coarse) ,
     $  alat_coarse(im_local_coarse,jm_local_coarse) ,
     $  mask_coarse(im_local_coarse,jm_local_coarse)
!_______________________________________________________________________
! 3-D arrays
      real
     $  aam            ,! horizontal kinematic viscosity
     $  advx           ,! x-horizontal advection and diffusion terms
     $  advy           ,! y-horizontal advection and diffusion terms
     $  a              ,
     $  c              ,
     $  drhox          ,! x-component of the internal baroclinic pressure
     $  drhoy          ,! y-component of the internal baroclinic pressure
     $  dtef           ,
     $  ee             ,
     $  gg             ,
     $  kh             ,! vertical diffusivity
     $  km             ,! vertical kinematic viscosity
     $  kq             ,
     $  l              ,! turbulence length scale
     $  q2b            ,! twice the turbulent kinetic energy at time n-1
     $  q2             ,! twice the turbulent kinetic energy at time n
     $  q2lb           ,! q2 x l at time n-1
     $  q2l            ,! q2 x l at time n
     $  rho            ,! density
     $  rmean          ,! horizontally averaged density
     $  sb             ,! salinity at time n-1
     $  sclim          ,! horizontally averaged salinity
     $  s              ,! salinity at time n
     $  tb             ,! temperature at time n-1
     $  tclim          ,! horizontally averaged temperature
     $  t              ,! temperature at time n
     $  ub             ,! horizontal velocity in x at time n-1
     $  uf             ,! horizontal velocity in x at time n+1
     $  u              ,! horizontal velocity in x at time n
     $  ustks          ,! Stokes velocity in x   !lyo:!stokes
     $  vb             ,! horizontal velocity in y at time n-1
     $  vf             ,! horizontal velocity in y at time n+1
     $  v              ,! horizontal velocity in y at time n
     $  vstks          ,! Stokes velocity in y   !lyo:!stokes
     $  w              ,! sigma coordinate vertical velocity
     $  wr             ,! real (z coordinate) vertical velocity
     $  xstks          ,! Stokes vortex+coriolis in x   !lyo:!stokes
     $  ystks          ,! Stokes vortex+coriolis in y   !lyo:!stokes
     $  zflux


      common/blk3d/
     $  aam(im_local,jm_local,kb)  ,
     $  advx(im_local,jm_local,kb) ,
     $  advy(im_local,jm_local,kb) ,
     $  a(im_local,jm_local,kb)    ,
     $  c(im_local,jm_local,kb)    ,
     $  drhox(im_local,jm_local,kb),
     $  drhoy(im_local,jm_local,kb),
     $  dtef(im_local,jm_local,kb) ,
     $  ee(im_local,jm_local,kb)   ,
     $  gg(im_local,jm_local,kb)   ,
     $  kh(im_local,jm_local,kb)   ,
     $  km(im_local,jm_local,kb)   ,
     $  kq(im_local,jm_local,kb)   ,
     $  l(im_local,jm_local,kb)    ,
     $  q2b(im_local,jm_local,kb)  ,
     $  q2(im_local,jm_local,kb)   ,
     $  q2lb(im_local,jm_local,kb) ,
     $  q2l(im_local,jm_local,kb)  ,
     $  rho(im_local,jm_local,kb)  ,
     $  rmean(im_local,jm_local,kb),
     $  sb(im_local,jm_local,kb)   ,
     $  sclim(im_local,jm_local,kb),
     $  s(im_local,jm_local,kb)    ,
     $  tb(im_local,jm_local,kb)   ,
     $  tclim(im_local,jm_local,kb),
     $  t(im_local,jm_local,kb)    ,
     $  ub(im_local,jm_local,kb)   ,
     $  uf(im_local,jm_local,kb)   ,
     $  u(im_local,jm_local,kb)    ,
     $  ustks(im_local,jm_local,kb),  !lyo:!stokes
     $  vb(im_local,jm_local,kb)   ,
     $  vf(im_local,jm_local,kb)   ,
     $  v(im_local,jm_local,kb)    ,
     $  vstks(im_local,jm_local,kb),  !lyo:!stokes
     $  w(im_local,jm_local,kb)    ,
     $  wr(im_local,jm_local,kb)   ,
     $  xstks(im_local,jm_local,kb),  !lyo:!stokes
     $  ystks(im_local,jm_local,kb),  !lyo:!stokes
     $  zflux(im_local,jm_local,kb)

! ================================================
! ayumi 2010/4/15 

      logical 
     $  calc_wind, calc_tsforce,
     $  calc_river, calc_assim,
     $  calc_assimdrf, !eda
     $  calc_trajdrf, !roger
     $  calc_interp,    !fhx:interp_flag
     $  calc_tsurf_mc,  !fhx:mcsst
     $  calc_tide,      !fhx:tide
     $  calc_stokes,    !lyo:stokes
     $  calc_vort       !lyo:!vort

      integer 
     $  num, iout

! 2-d
      real 
     $  uab_mean, vab_mean, elb_mean,
     $  wusurf_mean, wvsurf_mean,
     $  wtsurf_mean, wssurf_mean

! 3-d
      real
     $  u_mean, v_mean, w_mean,
     $  t_mean, s_mean, rho_mean,
     $  kh_mean, km_mean,
     $  xstks_mean, ystks_mean !lyo:!stokes:


      common/blklog/ 
     $  calc_wind, 
     $  calc_tsforce,
     $  calc_river,
     $  calc_assim,
     $  calc_assimdrf, !eda
     $  calc_trajdrf,  !roger
     $  calc_interp,    !fhx:interp_flag
     $  calc_tsurf_mc,  !fhx:mcsst
     $  calc_tide,      !fhx:tide
     $  calc_stokes,    !lyo:stokes
     $  calc_vort       !lyo:!vort

      common/blkcon2/
     $  num, iout          

      common/blk2d2/
     $  uab_mean(im_local,jm_local)    ,
     $  vab_mean(im_local,jm_local)    ,
     $  elb_mean(im_local,jm_local)    ,
     $  wusurf_mean(im_local,jm_local) ,
     $  wvsurf_mean(im_local,jm_local) ,
     $  wtsurf_mean(im_local,jm_local) ,
     $  wssurf_mean(im_local,jm_local)    

      common/blk3d2/
     $  u_mean(im_local,jm_local,kb)   ,
     $  v_mean(im_local,jm_local,kb)   ,
     $  w_mean(im_local,jm_local,kb)   ,
     $  t_mean(im_local,jm_local,kb)   , 
     $  s_mean(im_local,jm_local,kb)   ,
     $  rho_mean(im_local,jm_local,kb) ,
     $  kh_mean(im_local,jm_local,kb)  ,  
     $  km_mean(im_local,jm_local,kb)  ,  
     $  xstks_mean(im_local,jm_local,kb)  ,  !lyo:!stokes:
     $  ystks_mean(im_local,jm_local,kb)     !lyo:!stokes:




! ================================================


!_______________________________________________________________________
! 1 and 2-D boundary value arrays
      real
     $  ele            ,! elevation at the eastern open boundary
     $  eln            ,! elevation at the northern open boundary
     $  els            ,! elevation at the southern open boundary
     $  elw            ,! elevation at the western open boundary
     $  sbe            ,! salinity at the eastern open boundary
     $  sbn            ,! salinity at the northern open boundary
     $  sbs            ,! salinity at the southern open boundary
     $  sbw            ,! salinity at the western open boundary
     $  tbe            ,! temperature at the eastern open boundary
     $  tbn            ,! temperature at the northern open boundary
     $  tbs            ,! temperature at the southern open boundary
     $  tbw            ,! temperature at the western open boundary
     $  uabe           ,! vertical mean of u at the eastern open boundary
     $  uabw           ,! vertical mean of u at the western open boundary
     $  ube            ,! u at the eastern open boundary
     $  ubw            ,! u at the western open boundary
     $  vabn           ,! vertical mean of v at the northern open boundary
     $  vabs           ,! vertical mean of v at the southern open boundary
     $  vbn            ,! v at the northern open boundary
     $  vbs            ,! v at the southern open boundary
     $  ampe           ,! M2/K1 eta amplitude at the eastern open boundary !fhx:tide
     $  phae           ,! M2/K1 eta phase at the eastern open boundary     !fhx:tide
     $  amue           ,! M2/k1 UA amplitude at the eastern open boundary  !fhx:tide
     $  phue            ! M2/k1 UA phase at the eastern open boundary      !fhx:tide

      common/bdry1/     
     $  ele(jm_local)        ,
     $  eln(im_local)        ,
     $  els(im_local)        ,
     $  elw(jm_local)        ,
     $  sbe(jm_local,kb)     ,
     $  sbn(im_local,kb)     ,
     $  sbs(im_local,kb)     ,
     $  sbw(jm_local,kb)     ,
     $  tbe(jm_local,kb)     ,
     $  tbn(im_local,kb)     ,
     $  tbs(im_local,kb)     ,
     $  tbw(jm_local,kb)     ,
     $  uabe(jm_local)       ,
     $  uabw(jm_local)       ,
     $  ube(jm_local,kb)     ,
     $  ubw(jm_local,kb)     ,
     $  vabn(im_local)       ,
     $  vabs(im_local)       ,
     $  vbn(im_local,kb)     ,
     $  vbs(im_local,kb)     ,
     $  ampe(jm_local,ntide)       , !fhx:tide
     $  phae(jm_local,ntide)       , !fhx:tide
     $  amue(jm_local,ntide)       , !fhx:tide
     $  phue(jm_local,ntide)         !fhx:tide


!lyo:pac10:beg:
      integer
     $  nfe            ,! relax pnts next to bound e
     $  nfw            ,! relax pnts next to bound w
     $  nfn            ,! relax pnts next to bound n
     $  nfs             ! relax pnts next to bound s

      parameter(
     $  nfe=  0        ,
     $  nfw=  0        ,
     $  nfn=  0        ,
     $  nfs=  0         )

      real
     $  elobe           ,! elevation at the eastern open boundary
     $  elobw           ,! elevation at the western open boundary
     $  elobn           ,! elevation at the northern open boundary
     $  elobs           ,! elevation at the southern open boundary
     $  sobe            ,! salinity at the eastern open boundary
     $  sobw            ,! salinity at the western open boundary
     $  sobn            ,! salinity at the northern open boundary
     $  sobs            ,! salinity at the southern open boundary
     $  tobe            ,! temperature at the eastern open boundary
     $  tobw            ,! temperature at the western open boundary
     $  tobn            ,! temperature at the northern open boundary
     $  tobs            ,! temperature at the southern open boundary
     $  uaobe           ,! vertical mean of u at the eastern open boundary
     $  uaobw           ,! vertical mean of u at the western open boundary
     $  uaobn           ,! vertical mean of u at the northern open boundary
     $  uaobs           ,! vertical mean of u at the southern open boundary
     $  uobe            ,! u at the eastern open boundary
     $  uobw            ,! u at the western open boundary
     $  uobn            ,! u at the northern open boundary
     $  uobs            ,! u at the southern open boundary
     $  vaobe           ,! vertical mean of v at the eastern open boundary
     $  vaobw           ,! vertical mean of v at the western open boundary
     $  vaobn           ,! vertical mean of v at the northern open boundary
     $  vaobs           ,! vertical mean of v at the southern open boundary
     $  vobe            ,! v at the eastern open boundary
     $  vobw            ,! v at the western open boundary
     $  vobn            ,! v at the northern open boundary
     $  vobs            ,! v at the southern open boundary
     $  frz             ,! flow-relax-coefficient at boundaries
     $  aamfrz           ! sponge factor - increased aam at boundaries

      common/bdry/
     $  elobe(nfe,jm_local)       ,
     $  elobw(nfw,jm_local)       ,
     $  elobn(im_local,nfn)       ,
     $  elobs(im_local,nfs)       ,
     $  sobe(nfe,jm_local,kb)     ,
     $  sobw(nfw,jm_local,kb)     ,
     $  sobn(im_local,nfn,kb)     ,
     $  sobs(im_local,nfs,kb)     ,
     $  tobe(nfe,jm_local,kb)     ,
     $  tobw(nfw,jm_local,kb)     ,
     $  tobn(im_local,nfn,kb)     ,
     $  tobs(im_local,nfs,kb)     ,
     $  uaobe(nfe,jm_local)       ,
     $  uaobw(nfw,jm_local)       ,
     $  uaobn(im_local,nfn)       ,
     $  uaobs(im_local,nfs)       ,
     $  uobe(nfe,jm_local,kb)     ,
     $  uobw(nfw,jm_local,kb)     ,
     $  uobn(im_local,nfn,kb)     ,
     $  uobs(im_local,nfs,kb)     ,
     $  vaobe(nfe,jm_local)       ,
     $  vaobw(nfw,jm_local)       ,
     $  vaobn(im_local,nfn)       ,
     $  vaobs(im_local,nfs)       ,
     $  vobe(nfe,jm_local,kb)     ,
     $  vobw(nfw,jm_local,kb)     ,
     $  vobn(im_local,nfn,kb)     ,
     $  vobs(im_local,nfs,kb)     ,
     $  frz(im_local,jm_local)    ,
     $  aamfrz(im_local,jm_local) 
!lyo:pac10:end:

!_______________________________________________________________________
! Character variables
      character*4
     $  windf           !lyo: windf = ccmp, ecmw, or gfsw etc

      character*26
     $  time_start      ! date and time of start of initial run of model

      character*40
     $  source         ,
     $  title

      character*120
     $  netcdf_file    ,
     $  read_rst_file  ,
     $  write_rst_file

      common/blkchar/
     $  windf          ,!lyo: windf = ccmp, ecmw, or gfsw etc
     $  time_start     ,
     $  source         ,
     $  title          ,
     $  netcdf_file    ,
     $  read_rst_file  ,
     $  write_rst_file

!_______________________________________________________________________
! Logical variables
      logical lramp

      common/blklog/ lramp


! ================================================
      integer 
     $  output_flag, SURF_flag, !fhx:tracer: !fhx:20110131:
     $  tracer_flag !fhx:tracer:value = -2,-1,0,+1 or 2
                    ! =0: no tracer run;
                    ! >0: tracer run and read rstart w/  tracer
                    ! <0: tracer run but read rstart w/o tracer
                    !+-1: direct specification: tr = val
                    !+-2: source specification: d(tr)/dt = tr_source

!fhx:20110131:beg:
      integer 
     $  nums, iprints,iouts
! 2-d
      real 
     $  usrf_mean, vsrf_mean, elsrf_mean,
     $  uwsrf_mean, vwsrf_mean,uwsrf,vwsrf,
     $  utf_mean, vtf_mean !lyo:!vort:U-transport!part of CPVF:

      common/blkflag/ 
     $  output_flag, 
     $  SURF_flag,    !fhx:tracer:
     $  tracer_flag   !fhx:tracer:

      common/blk0dsurf/
     $  nums, iprints,iouts          

      common/blk2dsurf/
     $  usrf_mean(im_local,jm_local)    ,
     $  vsrf_mean(im_local,jm_local)    ,
     $  elsrf_mean(im_local,jm_local)   ,
     $  uwsrf_mean(im_local,jm_local)   ,
     $  vwsrf_mean(im_local,jm_local)   ,
     $  uwsrf(im_local,jm_local)        ,  !wind data for SURF mean output
     $  vwsrf(im_local,jm_local)        ,
     $  utf_mean(im_local,jm_local)     ,  !lyo:!vort:U-transport!part of CPVF:
     $  vtf_mean(im_local,jm_local)
!fhx:20110131:end:

!fhx:tracer:beg
      real dispc   ! "disposable" real variable
      integer  
     $  nb,        ! #tracers (e.g.=#release points)
     $  inb        ! "tracer loop" integer variable
      parameter(nb=1)
      common/blk0dtracer/ inb,dispc
!1-d arrays 
!     integer itr,jtr,ktr ! (i,j,k) where tracers are released
      integer :: itr( nb ) =
!    $(/ 198, 427  /) !Xiamen, Fukushima
!!   $(/ 198, 199  /) !Xiamen, Xiamen2
     $(/  20  /) !fake =(im_global/2,jm_global/2)
      integer :: jtr( nb ) =
!    $(/ 400, 532  /) !Xiamen, Fukushima
!!   $(/ 400, 398  /) !Xiamen, Xiamen2
     $(/  10  /) !fake
      integer :: ktr( nb ) =
!!   $(/  -1,  -1  /) !column tracer if ktr<0 --> k=1,kbm1,-ktr
     $(/  -1  /) !column tracer if ktr<0 --> k=1,kbm1,-ktr
!2-d arrays
      real rdisp2d
      common/blk2dtracer/
     $ rdisp2d(im_local,jm_local)
!3-d arrays
      real tr3db,tr3d
      common/blk3dtracer/
     $ tr3db(im_local,jm_local,kb),tr3d(im_local,jm_local,kb)
      real 
     $  tre, trn, trs, trw  ! boundary value arrays 
      common/bdry_tracer/
     $ tre(jm_local,kb,nb),
     $ trn(im_local,kb,nb), 
     $ trs(im_local,kb,nb),
     $ trw(jm_local,kb,nb)
!4-d arrays     
      real 
!w/o tr_mean to save space, but can be changed - search 'tr_mean'
!    $  trb, tr, tr_mean
     $  trb, tr
      common/blk4dtracer/
     $  trb(im_local,jm_local,kb,nb)       ,
     $  tr(im_local,jm_local,kb,nb)
!    $  tr(im_local,jm_local,kb,nb)        ,
!    $  tr_mean(im_local,jm_local,kb,nb)  
!fhx:tracer:end
!--------------------------------------------------------------------
! trajdrf variable !roger
      integer
     $  np                 ! number of particle drifter---roger
      
      real 
     $  xstart ,              ! alon for drifter position----roger
     $  ystart ,              ! alat for drifter position----roger
     $  UXA,                  !record uv velocity
     $  VXA

      common/traj/
     $  np          ,
     $  xstart(100)  ,         ! can not use allocate----use 30 spaces
     $  ystart(100)  ,         ! roger 
     $  UXA(im_local,jm_local),
     $  VXA(im_local,jm_local)
!--------------------------------------------------------------------
!lyo:vort:
      integer 
     $  ioutv
      common/blk0dvort/
     $  ioutv
      real 
     $  FX,FY,CTOT,CELG,CTSURF,CTBOT,CPVF,CJBAR,CADV,CTEN,TOTX,TOTY
     $ ,CELG_mean,CTSURF_mean,CTBOT_mean,CPVF_mean
     $ ,CJBAR_mean,CADV_mean,CTEN_mean

      common/vorta/
     $  FX(im_local,jm_local),FY(im_local,jm_local),
     $  CTOT(im_local,jm_local),CELG(im_local,jm_local),
     $  CTSURF(im_local,jm_local),CTBOT(im_local,jm_local),
     $  CPVF(im_local,jm_local),CJBAR(im_local,jm_local),
     $  CADV(im_local,jm_local),CTEN(im_local,jm_local),
     $  TOTX(im_local,jm_local),TOTY(im_local,jm_local)
     $ ,CELG_mean(im_local,jm_local),CTSURF_mean(im_local,jm_local)
     $ ,CTBOT_mean(im_local,jm_local),CPVF_mean(im_local,jm_local)
     $ ,CJBAR_mean(im_local,jm_local),CADV_mean(im_local,jm_local)
     $ ,CTEN_mean(im_local,jm_local)
!--------------------------------------------------------------------




