
!_______________________________________________________________________
      subroutine read_namelist_pom(title, netcdf_file, mode, nadv, 
     $                         nitera, sw, npg, dte, isplit, time_start,
     $                         nread_rst, read_rst_file, write_rst, 
     $                         write_rst_file, days, prtd1, prtd2, 
     $                         iperx, ipery, n1d, ngrid, windf, 
     $                         im_global, jm_global, kb,
     $                         im_local,jm_local, im_global_coarse,
     $                         jm_global_coarse, im_local_coarse,
     $                         jm_local_coarse, x_division,
     $                         y_division, n_proc)
     
! read input values and defines constants
      implicit none
      character*4   windf 
      character*26  time_start 
      character*40  title
      character*120 netcdf_file, read_rst_file, write_rst_file
      integer       mode, nadv, nitera, npg, isplit, nread_rst
      integer       im_global, jm_global, kb, im_local, jm_local, n_proc
      integer       im_global_coarse, jm_global_coarse
      integer       im_local_coarse, jm_local_coarse
      integer       x_division, y_division 
      integer       iperx, ipery, n1d, ngrid 
      real          sw, dte, write_rst, days, prtd1, prtd2 

      namelist/pom_nml/ title,netcdf_file,mode,nadv,nitera,sw,npg,dte,
     $                  isplit,time_start,nread_rst,read_rst_file,
     $                  write_rst,write_rst_file,days,prtd1,prtd2,
     $                  iperx, ipery, n1d, ngrid, windf, 
     $                  im_global,jm_global,kb,im_local,jm_local,
     $                  im_global_coarse, jm_global_coarse,
     $                  im_local_coarse, jm_local_coarse,
     $                  x_division, y_division, n_proc

! read input namelist
      open(73,file='pom.nml',status='old')
      read(73,nml=pom_nml)
      close(73)
      return
      end

      subroutine read_namelist_switch(
     $                     calc_wind, calc_tsforce,
     $                     calc_river, calc_assim,
     $                     calc_assimdrf, calc_tsurf_mc,
     $                     calc_tide, tracer_flag,
     $                     calc_stokes, calc_vort,
     $                     output_flag, SURF_flag)
     
! read input values and defines constants
      implicit none
      logical       calc_wind, calc_tsforce, calc_river,
     $              calc_assim, calc_assimdrf, calc_tsurf_mc,
     $              calc_tide, calc_trajdrf, calc_stokes, 
     $              calc_vort 

      integer       tracer_flag, output_flag, SURF_flag

      namelist/switch_nml/ calc_wind, calc_tsforce, calc_river,
     $                     calc_assim, calc_assimdrf, calc_tsurf_mc,
     $                     calc_tide, calc_trajdrf, 
     $                     tracer_flag,
     $                     calc_stokes, calc_vort,
     $                     output_flag, SURF_flag

! read input namelist
      open(73,file='switch.nml',status='old')
      read(73,nml=switch_nml)
      close(73)
      return
      end
