! Updated (in/grid/grid.nc & >2GB) io_pnetcdf.F combining FH+Alu
!     This version is for FH's case with AluUpperSST & AluUpperSSS
!     commented out
!
! io_pnetcdf.F !lyo:pac10:Incorporating Alu's version for pac10
!              !          in ncu's /home/lyo/gaea/alu/pac10

! parallel NetCDF I/O

!_______________________________________________________________________
      subroutine def_var_pnetcdf(ncid,name,nvdims,vdims,varid,
     $                           long_name,units,coords,lcoords)
# include "mpif.h"
# include "pnetcdf.inc"
      integer vdims(nvdims)
      integer ncid,nvdims,varid
      logical lcoords
      character*(*) name,long_name,units,coords
      integer status
      integer(mpi_offset_kind) length
! define variable
      status=nfmpi_def_var(ncid,name,nf_float,nvdims,vdims,varid)
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)
! define attributes
      length=len(trim(long_name))
      status=nfmpi_put_att_text(ncid,varid,'long_name',
     $                          length,trim(long_name))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      length=len(trim(units))
      status=nfmpi_put_att_text(ncid,varid,'units',
     $                          length,trim(units))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)
! add coordinates attribute, if necessary
      if(lcoords) then
        length=len(trim(coords))
        status=nfmpi_put_att_text(ncid,varid,'coordinates',
     $                            length,trim(coords))
        call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)
      endif
      return
      end

!_______________________________________________________________________
      subroutine handle_error_pnetcdf(routine,status,nf_noerr)
      implicit none
      include 'pom.h'
      integer status,nf_noerr
      character*(*) routine
      if(status.ne.nf_noerr) then
        error_status=1
        if(my_task.eq.master_task) then
           write(*,'(/a,a)')  'Error: NetCDF routine ',routine
           stop
        endif
      end if
      return
      end

!_______________________________________________________________________
!      subroutine write_output_pnetcdf
! ayumi 2010/5/10
      subroutine write_output_pnetcdf0( netcdf_out_file )
! write output data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer nprint
      integer time_dimid,x_dimid,y_dimid,z_dimid,nb_dimid !fhx:tracer
      integer z_varid,zz_varid,dx_varid,dy_varid,east_c_varid,
     $        east_e_varid,east_u_varid,east_v_varid,north_c_varid,
     $        north_e_varid,north_u_varid,north_v_varid,rot_varid,
     $        h_varid,fsm_varid,dum_varid,dvm_varid,
     $        wusurf_varid,wvsurf_varid,wtsurf_varid,wssurf_varid
      integer time_varid,uab_varid,vab_varid,elb_varid,u_varid,v_varid,
     $        w_varid,t_varid,s_varid,rho_varid,km_varid,kh_varid
      integer tr_varid  !fhx:tracer
      integer ncid,status
      integer vdims(5) !fhx:tracer
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(5),edge(5) !fhx:tracer
      

! create netcdf file
!      nprint=(iint+time0*86400/dti)/iprint
!      write(netcdf_out_file,'(''out/'',a,''.'',i4.4,''.nc'')') 
!     $                                          trim(netcdf_file),nprint
      if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $     trim(netcdf_out_file)
      status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
      call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=1
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)
!fhx:tracer:begin
      length=nb
      status=nfmpi_def_dim(ncid,'nb',length,nb_dimid)
      call handle_error_pnetcdf('nf_def_dim: nb',status,nf_noerr)
!fhx:tracer:end

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid
      call def_var_pnetcdf(ncid,'dx',2,vdims,dx_varid,
     $                     'grid increment in x','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'dy',2,vdims,dy_varid,
     $                     'grid increment in y','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'east_u',2,vdims,east_u_varid,
     $                     'easting of u-points','degree',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'east_v',2,vdims,east_v_varid,
     $                     'easting of v-points','degree',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'east_e',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'east_c',2,vdims,east_c_varid,
     $                     'easting of cell corners','degree',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'north_u',2,vdims,north_u_varid,
     $                     'northing of u-points','degree',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'north_v',2,vdims,north_v_varid,
     $                     'northing of v-points','degree',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'north_e',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'north_c',2,vdims,north_c_varid,
     $                     'northing of cell corners','degree',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'rot',2,vdims,rot_varid,
     $                    'Rotation angle of x-axis wrt. east','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'dum',2,vdims,dum_varid,
     $                     'u-velocity mask','dimensionless',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'dvm',2,vdims,dvm_varid,
     $                     'v-velocity mask','dimensionless',
     $                     'east_v north_v',.true.)
     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'uab',3,vdims,uab_varid,
     $                     'depth-averaged u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vab',3,vdims,vab_varid,
     $                     'depth-averaged v','metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'elb',3,vdims,elb_varid,
     $                     'surface elevation','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'wusurf',3,vdims,wusurf_varid,
     $                     'x-momentum flux','metre^2/sec^2',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'wvsurf',3,vdims,wvsurf_varid,
     $                     'y-momentum flux','metre^2/sec^2',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'wtsurf',3,vdims,wtsurf_varid,
     $                     'temperature flux','deg m/s',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'wssurf',3,vdims,wssurf_varid,
     $                     'salinity flux','psu m/s',
     $                     'east_e north_e',.true.)
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=time_dimid
      call def_var_pnetcdf(ncid,'u',4,vdims,u_varid,
     $                     'x-velocity','metre/sec',
     $                     'east_u north_u zz',.true.)
      call def_var_pnetcdf(ncid,'v',4,vdims,v_varid,
     $                     'y-velocity','metre/sec',
     $                     'east_v north_v zz',.true.)
      call def_var_pnetcdf(ncid,'w',4,vdims,w_varid,
     $                     'z-velocity','metre/sec',
     $                     'east_e north_e z',.true.)
      call def_var_pnetcdf(ncid,'t',4,vdims,t_varid,
     $                     'potential temperature','K',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'s',4,vdims,s_varid,
     $                     'salinity x rho / rhoref','PSS',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'rho',4,vdims,rho_varid,
     $                     '(density-1000)/rhoref','dimensionless',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'kh',4,vdims,kh_varid,
     $                     'vertical diffusivity','metre2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'km',4,vdims,km_varid,
     $                     'vertical viscosity','metre2/sec',
     $                     'east_e north_e zz',.true.)

!fhx:tracer:begin
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=nb_dimid
      vdims(5)=time_dimid
      if (abs(tracer_flag).eq.1) then 

      call def_var_pnetcdf(ncid,'tr',5,vdims,tr_varid,
     $                     'tracer(0-1)','dimensionless',
     $                     'east_u north_u zz nb',.true.)
      else
      call def_var_pnetcdf(ncid,'tr',5,vdims,tr_varid,
     $                     'tracer(0~1e-5)','dimensionless',
     $                     'east_u north_u zz nb',.true.)
      endif

!fhx:tracer:end
! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
      start(1)=1
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_put_vara_real_all(ncid,dx_varid,start,edge,dx)
      call handle_error_pnetcdf('nf_put_var_real: dx',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dy_varid,start,edge,dy)
      call handle_error_pnetcdf('nf_put_var_real: dy',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_u_varid,start,edge,
     $                               east_u)
      call handle_error_pnetcdf('nf_put_var_real: east_u',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_v_varid,start,edge,
     $                               east_v)
      call handle_error_pnetcdf('nf_put_var_real: east_v',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_c_varid,start,edge,
     $                               east_c)
      call handle_error_pnetcdf('nf_put_var_real: east_c',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_u_varid,start,edge,
     $                               north_u)
      call handle_error_pnetcdf('nf_put_var_real: north_u',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_v_varid,start,edge,
     $                               north_v)
      call handle_error_pnetcdf('nf_put_var_real: north_v',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_c_varid,start,edge,
     $                               north_c)
      call handle_error_pnetcdf('nf_put_var_real: north_c',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rot_varid,start,edge,rot)
      call handle_error_pnetcdf('nf_put_var_real: rot',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dum_varid,start,edge,dum)
      call handle_error_pnetcdf('nf_put_var_real: dum',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dvm_varid,start,edge,dvm)
      call handle_error_pnetcdf('nf_put_var_real: dvm',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(ncid,uab_varid,start,edge,uab_mean)
      call handle_error_pnetcdf('nf_put_vara_real: uab',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,vab_varid,start,edge,vab_mean)
      call handle_error_pnetcdf('nf_put_vara_real: vab',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,elb_varid,start,edge,elb_mean)
      call handle_error_pnetcdf('nf_put_vara_real: elb',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wusurf_varid,start,edge,wusurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wusurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wvsurf_varid,start,edge,wvsurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wvsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wtsurf_varid,start,edge,wtsurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wtsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wssurf_varid,start,edge,wssurf_mean)
      call handle_error_pnetcdf('
     $     nf_put_vara_real: wssurf',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=1
      status=nfmpi_put_vara_real_all(ncid,u_varid,start,edge,u_mean)
      call handle_error_pnetcdf('nf_put_vara_real: u',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,v_varid,start,edge,v_mean)
      call handle_error_pnetcdf('nf_put_vara_real: v',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,w_varid,start,edge,w_mean)
      call handle_error_pnetcdf('nf_put_vara_real: w',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,t_varid,start,edge,t_mean)
      call handle_error_pnetcdf('nf_put_vara_real: t',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,s_varid,start,edge,s_mean)
      call handle_error_pnetcdf('nf_put_vara_real: s',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rho_varid,start,edge,rho_mean)
      call handle_error_pnetcdf('nf_put_vara_real: rho',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,kh_varid,start,edge,kh_mean)
      call handle_error_pnetcdf('nf_put_vara_real: kh',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,km_varid,start,edge,km_mean)
      call handle_error_pnetcdf('nf_put_vara_real: km',status,nf_noerr)
!fhx:tracer:begin
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      start(5)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=nb
      edge(5)=1
!     status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr_mean)
      status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr)
      call handle_error_pnetcdf('nf_put_vara_real: tr',status,nf_noerr)
!fhx:tracer:end
! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________
!      subroutine write_restart_pnetcdf
      subroutine write_restart_pnetcdf( netcdf_out_file )
! write data for a seamless restart
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp
      integer nprint
      integer ncid,status
      integer time_dimid,x_dimid,y_dimid,z_dimid,nb_dimid !fhx:tracer
      integer time_varid,wubot_varid,wvbot_varid,aam2d_varid,ua_varid,
     $        wusurf_varid, wvsurf_varid,
     $        uab_varid,va_varid,vab_varid,el_varid,elb_varid,et_varid,
     $        etb_varid,egb_varid,utb_varid,vtb_varid,u_varid,ub_varid,
     $        w_varid,v_varid,vb_varid,t_varid,tb_varid,s_varid,
     $        sb_varid,rho_varid,adx2d_varid,ady2d_varid,advua_varid,
     $        advva_varid,km_varid,kh_varid,kq_varid,l_varid,q2_varid,
     $        q2b_varid,aam_varid,q2l_varid,q2lb_varid, !fhx:tracer
     $        tr_varid,trb_varid  !fhx:tracer
      integer vdims(4) !fhx:tracer
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4) !fhx:tracer

! create netcdf restart file
      nprint=(iint+time0*86400/dti)/irestart
!      write(netcdf_out_file,'(''out/'',a,''.'',i4.4,''.nc'')') 
!     $     trim(write_rst_file),nprint
      if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $ trim(netcdf_out_file)
!      status=nfmpi_create(pom_comm,netcdf_out_file,nf_64bit_offset,
      status=nfmpi_create(pom_comm,netcdf_out_file,
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
      call handle_error_pnetcdf('nf_create',status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='restart file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=1
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim',status,nf_noerr)

      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim',status,nf_noerr)

      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim',status,nf_noerr)

      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim',status,nf_noerr)
!fhx:tracer:begin
      length=nb
      status=nfmpi_def_dim(ncid,'nb',length,nb_dimid)
      call handle_error_pnetcdf('nf_def_dim: nb',status,nf_noerr)
!fhx:tracer:end

! define variables and their attributes:
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,' ',.false.)

      vdims(1)=x_dimid
      vdims(2)=y_dimid
      call def_var_pnetcdf(ncid,'wusurf',2,vdims,wusurf_varid,
     $                     'x-momentum flux at the surface',
     $                     'metre^2/sec^2',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'wvsurf',2,vdims,wvsurf_varid,
     $                     'y-momentum flux at the surface',
     $                     'metre^2/sec^2',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'wubot',2,vdims,wubot_varid,
     $                     'x-momentum flux at the bottom',
     $                     'metre^2/sec^2',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'wvbot',2,vdims,wvbot_varid,
     $                     'y-momentum flux at the bottom',
     $                     'metre^2/sec^2',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'aam2d',2,vdims,aam2d_varid,
     $                     'vertical average of aam',
     $                     'metre^2/sec',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'ua',2,vdims,ua_varid,
     $                     'vertical mean of u',
     $                     'metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'uab',2,vdims,uab_varid,
     $                     'vertical mean of u at time -dt',
     $                     'metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'va',2,vdims,va_varid,
     $                     'vertical mean of v',
     $                     'metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'vab',2,vdims,vab_varid,
     $                     'vertical mean of v at time -dt',
     $                     'metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'el',2,vdims,el_varid,
     $                     'surface elevation in external mode',
     $                     'metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'elb',2,vdims,elb_varid,
     $                     'surface elevation in external mode at -dt',
     $                     'metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'et',2,vdims,et_varid,
     $                     'surface elevation in internal mode',
     $                     'metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'etb',2,vdims,etb_varid,
     $                     'surface elevation in internal mode at -dt',
     $                     'metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'egb',2,vdims,egb_varid,
     $                     'surface elevation for pres. grad. at -dt',
     $                     'metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'utb',2,vdims,utb_varid,
     $                     'ua time averaged over dti',
     $                     'metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vtb',2,vdims,vtb_varid,
     $                     'va time averaged over dti',
     $                     'metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'adx2d',2,vdims,adx2d_varid,
     $                     'vertical integral of advx',
     $                     '-',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'ady2d',2,vdims,ady2d_varid,
     $                     'vertical integral of advy',
     $                     '-',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'advua',2,vdims,advua_varid,
     $                     'sum of 2nd, 3rd and 4th terms in eq (18)',
     $                     '-',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'advva',2,vdims,advva_varid,
     $                     'sum of 2nd, 3rd and 4th terms in eq (19)',
     $                     '-',
     $                     'east_v north_v',.true.)

      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      call def_var_pnetcdf(ncid,'u',3,vdims,u_varid,
     $                     'x-velocity',
     $                     'metre/sec',
     $                     'east_u north_u zz',.true.)
      call def_var_pnetcdf(ncid,'ub',3,vdims,ub_varid,
     $                     'x-velocity at time -dt',
     $                     'metre/sec',
     $                     'east_u north_u zz',.true.)
      call def_var_pnetcdf(ncid,'v',3,vdims,v_varid,
     $                     'y-velocity',
     $                     'metre/sec',
     $                     'east_v north_v zz',.true.)
      call def_var_pnetcdf(ncid,'vb',3,vdims,vb_varid,
     $                     'y-velocity at time -dt',
     $                     'metre/sec',
     $                     'east_v north_v zz',.true.)
      call def_var_pnetcdf(ncid,'w',3,vdims,w_varid,
     $                     'sigma-velocity',
     $                     'metre/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'t',3,vdims,t_varid,
     $                     'potential temperature',
     $                     'K',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'tb',3,vdims,tb_varid,
     $                     'potential temperature at time -dt',
     $                     'K',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'s',3,vdims,s_varid,
     $                     'salinity x rho / rhoref',
     $                     'PSS',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'sb',3,vdims,sb_varid,
     $                     'salinity x rho / rhoref at time -dt',
     $                     'PSS',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'rho',3,vdims,rho_varid,
     $                     '(density-1000)/rhoref',
     $                     'dimensionless',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'km',3,vdims,km_varid,
     $                     'vertical kinematic viscosity',
     $                     'metre^2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'kh',3,vdims,kh_varid,
     $                     'vertical diffusivity',
     $                     'metre^2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'kq',3,vdims,kq_varid,
     $                     'kq',
     $                     'metre^2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'l',3,vdims,l_varid,
     $                     'turbulence length scale',
     $                     '-',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'q2',3,vdims,q2_varid,
     $                     'twice the turbulent kinetic energy',
     $                     'metre^2/sec^2',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'q2b',3,vdims,q2b_varid,
     $                     'twice the turbulent kinetic energy at -dt',
     $                     'metre^2/sec^2',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'aam',3,vdims,aam_varid,
     $                     'horizontal kinematic viscosity',
     $                     'metre^2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'q2l',3,vdims,q2l_varid,
     $                     'q2 x l',
     $                     'metre^3/sec^2',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'q2lb',3,vdims,q2lb_varid,
     $                     'q2 x l at time -dt',
     $                     'metre^3/sec^2',
     $                     'east_e north_e zz',.true.)

!fhx:tracer:begin
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=nb_dimid
      call def_var_pnetcdf(ncid,'tr',4,vdims,tr_varid,
     $                     'tracer ','dimensionless',
     $                     'east_u north_u zz nb',.true.)
      call def_var_pnetcdf(ncid,'trb',4,vdims,trb_varid,
     $                     'tracer ','dimensionless',
     $                     'east_u north_u zz nb',.true.)
!fhx:tracer:end

! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef',status,nf_noerr)

! write data
      start(1)=1
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_put_vara_real_all(
     $     ncid,wusurf_varid,start,edge,wusurf)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wvsurf_varid,start,edge,wvsurf)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,wubot_varid,start,edge,wubot)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,wvbot_varid,start,edge,wvbot)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,aam2d_varid,start,edge,aam2d)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,ua_varid,start,edge,ua)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,uab_varid,start,edge,uab)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,va_varid,start,edge,va)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,vab_varid,start,edge,vab)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,el_varid,start,edge,el)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,elb_varid,start,edge,elb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,et_varid,start,edge,et)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,etb_varid,start,edge,etb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,egb_varid,start,edge,egb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,utb_varid,start,edge,utb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,vtb_varid,start,edge,vtb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,adx2d_varid,start,edge,adx2d)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,ady2d_varid,start,edge,ady2d)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,advua_varid,start,edge,advua)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,advva_varid,start,edge,advva)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      status=nfmpi_put_vara_real_all(ncid,u_varid,start,edge,u)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,ub_varid,start,edge,ub)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,v_varid,start,edge,v)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,vb_varid,start,edge,vb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,w_varid,start,edge,w)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,t_varid,start,edge,t)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,tb_varid,start,edge,tb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,s_varid,start,edge,s)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,sb_varid,start,edge,sb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rho_varid,start,edge,rho)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,km_varid,start,edge,km)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,kh_varid,start,edge,kh)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,kq_varid,start,edge,kq)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,l_varid,start,edge,l)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,q2_varid,start,edge,q2)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,q2b_varid,start,edge,q2b)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,aam_varid,start,edge,aam)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,q2l_varid,start,edge,q2l)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,q2lb_varid,start,edge,q2lb)
      call handle_error_pnetcdf('nf_put_vara_real',status,nf_noerr)

!fhx:tracer:begin
      if (tracer_flag.ne.0) then
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=nb

      status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr)
      call handle_error_pnetcdf('nf_put_vara_real: tr',status,nf_noerr)

      status=nfmpi_put_vara_real_all(ncid,trb_varid,start,edge,trb)
      call handle_error_pnetcdf('nf_put_vara_real: tr',status,nf_noerr)
      endif
!fhx:tracer:end

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end

!_______________________________________________________________________
      subroutine read_grid_pnetcdf
! read grid data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character*120 netcdf_grid_file
      integer z_varid,zz_varid,dx_varid,dy_varid,east_c_varid,
     $        east_e_varid,east_u_varid,east_v_varid,north_c_varid,
     $        north_e_varid,north_u_varid,north_v_varid,rot_varid,
     $        h_varid,fsm_varid,dum_varid,dvm_varid
      integer ncid,status
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)

! open netcdf file
      write(netcdf_grid_file,'(a)') "./in/grid/grid.nc"
   
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_grid_file)
      status=nfmpi_open(pom_comm,netcdf_grid_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_grid_file,
     $                          status,nf_noerr)
     
! get variables
      status=nfmpi_inq_varid(ncid,'z',z_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: z'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'zz',zz_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: zz'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'dx',dx_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: dx'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'dy',dy_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: dy'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'east_u',east_u_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: east_u'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'east_v',east_v_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: east_v'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'east_e',east_e_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: east_e'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'east_c',east_c_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: east_c'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'north_u',north_u_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: north_u'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'north_v',north_v_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: north_v'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'north_e',north_e_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: north_e'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'north_c',north_c_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: north_c'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'rot',rot_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: rot'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'h',h_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: h'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'fsm',fsm_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: fsm'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'dum',dum_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: dum'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'dvm',dvm_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: dvm'
     $                          ,status,nf_noerr)
! get data
      start(1)=1
      edge(1)=kb
      status=nfmpi_get_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_get_vara_real_all(ncid,dx_varid,start,edge,dx)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,dy_varid,start,edge,dy)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,east_u_varid,start,edge,
     $                               east_u)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,east_v_varid,start,edge,
     $                               east_v)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,east_c_varid,start,edge,
     $                               east_c)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,north_u_varid,start,edge,
     $                               north_u)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,north_v_varid,start,edge,
     $                               north_v)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,north_c_varid,start,edge,
     $                               north_c)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,rot_varid,start,edge,rot)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,dum_varid,start,edge,dum)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,dvm_varid,start,edge,dvm)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file:
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: grid',status,nf_noerr)

      return
      end
!_______________________________________________________________________
      subroutine read_grid_pnetcdf0  ! fhx:called by subroutine "interp_init"  
! read original grid data for wind and satellite data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character*120 netcdf_grid_file
      integer alon_coarse_varid,alat_coarse_varid,mask_coarse_varid
      integer ncid,status
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)

! open netcdf file, /wrk/newshoni/fayumi/work/mpi-profs/netcdf_files/grid/grid_i402j252_hmin12m.nc
      write(netcdf_grid_file,'(a)') "./in/grid_i402j252_hmin12m.nc"
   
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_grid_file)
      status=nfmpi_open(pom_comm_coarse,netcdf_grid_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_grid_file,
     $                          status,nf_noerr)
     
! get variables

      status=nfmpi_inq_varid(ncid,'east_e',alon_coarse_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: east_e'
     $                          ,status,nf_noerr)

      status=nfmpi_inq_varid(ncid,'north_e',alat_coarse_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: north_e'
     $                          ,status,nf_noerr)

      status=nfmpi_inq_varid(ncid,'fsm',mask_coarse_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: fsm'
     $                          ,status,nf_noerr)

! get data


      start(1)=i_global_coarse(1)
      start(2)=j_global_coarse(1)
      edge(1)=im_coarse
      edge(2)=jm_coarse

      status=nfmpi_get_vara_real_all(ncid,alon_coarse_varid,start,edge,
     $                               alon_coarse)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      status=nfmpi_get_vara_real_all(ncid,alat_coarse_varid,start,edge,
     $                               alat_coarse)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      status=nfmpi_get_vara_real_all(ncid,mask_coarse_varid,start,edge,
     $                               mask_coarse)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)


! close file:
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: grid',status,nf_noerr)

      return
      end

!_______________________________________________________________________
      subroutine read_restart_pnetcdf
! read data for a seamless restart
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character*120 netcdf_in_file
      integer ncid,status
      integer time_varid,wubot_varid,wvbot_varid,aam2d_varid,ua_varid,
     $        uab_varid,va_varid,vab_varid,el_varid,elb_varid,et_varid,
     $        etb_varid,egb_varid,utb_varid,vtb_varid,u_varid,ub_varid,
     $        w_varid,v_varid,vb_varid,t_varid,tb_varid,s_varid,
     $        sb_varid,rho_varid,adx2d_varid,ady2d_varid,advua_varid,
     $        advva_varid,km_varid,kh_varid,kq_varid,l_varid,q2_varid,
     $        q2b_varid,aam_varid,q2l_varid,q2lb_varid, !fhx:tracer
     $        tr_varid,trb_varid !fhx:tracer
! ayumi 2010/4/8 
      integer wusurf_varid, wvsurf_varid 
      integer vdims(4) !fhx:tracer
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4) !fhx:tracer
      integer i,j

! open netcdf restart file
      write(netcdf_in_file,'(''in/'',a)') trim(read_rst_file)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $ trim(netcdf_in_file)
      status=nfmpi_open(pom_comm,netcdf_in_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//trim(read_rst_file),
     $                          status,nf_noerr)

! get variables
! ayumi 2010/4/8 --------------------------------------------
!      status=nfmpi_inq_varid(ncid,'time',time_varid)
!      call handle_error_pnetcdf('nfmpi_inq_varid: time'
!     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wusurf',wusurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wusurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wvsurf',wvsurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wvsurf'
     $                          ,status,nf_noerr)
!------------------------------------------------------------
      status=nfmpi_inq_varid(ncid,'wubot',wubot_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wubot'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wvbot',wvbot_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wvbot'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'aam2d',aam2d_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: aam2d'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ua',ua_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ua'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'uab',uab_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uab'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'va',va_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: va'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vab',vab_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vab'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'el',el_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: el'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'elb',elb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: elb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'et',et_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: et'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'etb',etb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: etb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'egb',egb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: egb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'utb',utb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: utb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vtb',vtb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vtb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'u',u_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: u'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ub',ub_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ub'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'v',v_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: v'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vb',vb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'w',w_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: w'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'t',t_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: t'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'tb',tb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'s',s_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: s'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sb',sb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: sb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'rho',rho_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: rho'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'adx2d',adx2d_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: adx2d'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ady2d',ady2d_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ady2d'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'advua',advua_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: advua'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'advva',advva_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: advva'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'km',km_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: km'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'kh',kh_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: kh'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'kq',kq_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: kq'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'l',l_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: l'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'q2',q2_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: q2'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'q2b',q2b_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: q2b'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'aam',aam_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: aam'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'q2l',q2l_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: q2l'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'q2lb',q2lb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: q2lb'
     $                          ,status,nf_noerr)
!fhx:tracer:begin
      if (tracer_flag.gt.0) then !lyo:tracer:source:
      status=nfmpi_inq_varid(ncid,'tr',tr_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tr'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'trb',trb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: trb'
     $                          ,status,nf_noerr)
      endif
!fhx:tracer:end

! get data
! ayumi 2010/4/8
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_get_vara_real_all(ncid,time_varid,start,edge,time0)
!      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
!     $                          status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
! ayumi 2010/4/8 --------------------------------------------------------
      status=nfmpi_get_vara_real_all(ncid,
     $                           wusurf_varid,start,edge,wusurf)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,
     $                           wvsurf_varid,start,edge,wvsurf)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
!------------------------------------------------------------------------
      status=nfmpi_get_vara_real_all(ncid,wubot_varid,start,edge,wubot)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,wvbot_varid,start,edge,wvbot)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,aam2d_varid,start,edge,aam2d)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ua_varid,start,edge,ua)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,uab_varid,start,edge,uab)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,va_varid,start,edge,va)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,vab_varid,start,edge,vab)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,el_varid,start,edge,el)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,elb_varid,start,edge,elb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,et_varid,start,edge,et)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,etb_varid,start,edge,etb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,egb_varid,start,edge,egb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,utb_varid,start,edge,utb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,vtb_varid,start,edge,vtb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,adx2d_varid,start,edge,adx2d)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ady2d_varid,start,edge,ady2d)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,advua_varid,start,edge,advua)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,advva_varid,start,edge,advva)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      status=nfmpi_get_vara_real_all(ncid,u_varid,start,edge,u)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ub_varid,start,edge,ub)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,v_varid,start,edge,v)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,vb_varid,start,edge,vb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,w_varid,start,edge,w)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,t_varid,start,edge,t)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,tb_varid,start,edge,tb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,s_varid,start,edge,s)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sb_varid,start,edge,sb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,rho_varid,start,edge,rho)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,km_varid,start,edge,km)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,kh_varid,start,edge,kh)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,kq_varid,start,edge,kq)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,l_varid,start,edge,l)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,q2_varid,start,edge,q2)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,q2b_varid,start,edge,q2b)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,aam_varid,start,edge,aam)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,q2l_varid,start,edge,q2l)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,q2lb_varid,start,edge,q2lb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
!fhx:tracer:begin
      if (tracer_flag.gt.0) then !lyo:tracer:source:
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=nb
      status=nfmpi_get_vara_real_all(ncid,tr_varid,start,edge,tr)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,trb_varid,start,edge,trb)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      endif
!fhx:tracer:end

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

! initialize elevation
      do j=1,jm
        do i=1,im
          d(i,j)=h(i,j)+el(i,j)
          dt(i,j)=h(i,j)+et(i,j)
        end do
      end do

! initialize time
      time=time0

      return
      end

!_______________________________________________________________________
      subroutine read_initial_ts_pnetcdf(k,temp,salt)
! read initial temperature and salinity
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer k
!      real temp(im,jm,k),salt(im,jm,k)
      real temp(im,jm,kb),salt(im,jm,kb)
      character*120 netcdf_ic_file
      integer ncid,status
      integer tb_varid,sb_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)

! open netcdf ic file
      write(netcdf_ic_file,'(a)') "./in/tsclim/ts_initial.nc"

      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_ic_file)
      status=nfmpi_open(pom_comm,netcdf_ic_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_ic_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'tb',tb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: t'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sb',sb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: s'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      status=nfmpi_get_vara_real_all(ncid,tb_varid,start,edge,temp)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sb_varid,start,edge,salt)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________
      subroutine read_clim_ts_pnetcdf(k,temp,salt)
! read annual-mean temperature and salinity
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer k
!      real temp(im,jm,k),salt(im,jm,k)
      real temp(im,jm,kb),salt(im,jm,kb)
      character*120 netcdf_ic_file
      integer ncid,status
      integer tclim_varid,sclim_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)

! open netcdf ic file
      write(netcdf_ic_file,'(a)') "./in/tsclim/ts_clim.nc"

      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_ic_file)
      status=nfmpi_open(pom_comm,netcdf_ic_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_ic_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'tclim',tclim_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: t'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sclim',sclim_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: s'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      status=nfmpi_get_vara_real_all(ncid,tclim_varid,start,edge,temp)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sclim_varid,start,edge,salt)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end

!_______________________________________________________________________
      subroutine read_mean_ts_pnetcdf(k,temp,salt)  !lyo:20110202
! read xy-averaged, annual-mean temp & salt

      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer k
!      real temp(im,jm,k),salt(im,jm,k)
      real temp(im,jm,kb),salt(im,jm,kb)
      character*120 netcdf_ic_file
      integer ncid,status
      integer tmean_varid,smean_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)
!
! open netcdf ic file !lyo:try first *mean* - the correct one
      write(netcdf_ic_file,'(a)') "./in/tsclim/ts_mean.nc"

      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_ic_file)
      status=nfmpi_open(pom_comm,netcdf_ic_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_ic_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'tmean',tmean_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: t'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'smean',smean_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: s'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      status=nfmpi_get_vara_real_all(ncid,tmean_varid,start,edge,temp)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,smean_varid,start,edge,salt)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end

!_______________________________________________________________________
      subroutine read_boundary_conditions_pnetcdf !lyo:pac10:not used!
! read boundary conditions
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character*120 netcdf_bc_file
      integer ncid,status
      integer tw_varid,sw_varid,uw_varid,ew_varid
      integer te_varid,se_varid,ue_varid,ee_varid
      integer ts_varid,ss_varid,vs_varid,es_varid
      integer tn_varid,sn_varid,vn_varid,en_varid
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)

! open netcdf ic file
      write(netcdf_bc_file,'(''in/'',a,''.bc.nc'')')
     $  trim(netcdf_file)
   
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_bc_file)
      status=nfmpi_open(pom_comm,netcdf_bc_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_bc_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'tw',tw_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tw',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sw',sw_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: sw',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'uw',uw_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uw',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ew',ew_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ew',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'te',te_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: te',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'se',se_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: se',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ue',ue_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ue',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ee',ee_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ee',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ts',ts_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ts',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ss',ss_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ss',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vs',vs_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vs',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'es',es_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: es',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'tn',tn_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tn',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sn',sn_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: sn',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vn',vn_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vn',status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'en',en_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: en',status,nf_noerr)

! get data
      start(1)=j_global(1)
      edge(1)=jm
      status=nfmpi_get_vara_real_all(ncid,uw_varid,start,edge,uabw)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: uabw',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ew_varid,start,edge,elw)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: elw',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ue_varid,start,edge,uabe)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: uabe',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ee_varid,start,edge,ele)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: ele',
     $                          status,nf_noerr)

      start(1)=i_global(1)
      edge(1)=im
      status=nfmpi_get_vara_real_all(ncid,vs_varid,start,edge,vabs)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: vabs',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,es_varid,start,edge,els)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: els',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,vn_varid,start,edge,vabn)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: vabn',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,en_varid,start,edge,eln)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: eln',
     $                          status,nf_noerr)

      start(1)=j_global(1)
      start(2)=1
      edge(1)=jm
      edge(2)=kb
      status=nfmpi_get_vara_real_all(ncid,tw_varid,start,edge,tbw)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: tbw',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sw_varid,start,edge,sbw)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: sbw',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,te_varid,start,edge,tbe)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: tbe',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,se_varid,start,edge,sbe)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: sbe',
     $                          status,nf_noerr)

      start(1)=i_global(1)
      start(2)=1
      edge(1)=im
      edge(2)=kb
      status=nfmpi_get_vara_real_all(ncid,ts_varid,start,edge,tbs)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: tbs',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,ss_varid,start,edge,sbs)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: sbs',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,tn_varid,start,edge,tbn)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: tbn',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sn_varid,start,edge,sbn)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all: sbn',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________
      subroutine read_wind_pnetcdf0(n,wu,wv) !lyo:pac10:not used!
! read wind stress
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer n
      real wu(im,jm),wv(im,jm)
      integer i,j
      character*120 netcdf_wind_file
      integer ncid,status
      integer wu_varid,wv_varid
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)

! open netcdf wind file
      write(netcdf_wind_file,'(''in/'',a,''.wind.'',i4.4,''.nc'')')
     $                                               trim(netcdf_file),n
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                            trim(netcdf_wind_file)
      status=nfmpi_open(pom_comm,netcdf_wind_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_wind_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'wu',wu_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wu'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wv',wv_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wv'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_get_vara_real_all(ncid,wu_varid,start,edge,wu)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,wv_varid,start,edge,wv)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________




! followings are for MAB model 
! ayumi since 2010/4/?
!_______________________________________________________________________
      subroutine read_tsclim_monthly_pnetcdf(tm,sm,tc,sc,in_file)
! read initial temperature and salinity
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer k
      real tm(im,jm,kb),sm(im,jm,kb),tc(im,jm,kb),sc(im,jm,kb)
      character*120 netcdf_ic_file
      character*13 in_file
      integer ncid,status
      integer tm_varid,sm_varid,tc_varid,sc_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)

! open netcdf ic file
      write(netcdf_ic_file,'(''in/tsclim/'',a)')
     $     trim(in_file)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_ic_file)
      status=nfmpi_open(pom_comm,netcdf_ic_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_ic_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'tm',tm_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tm'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sm',sm_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: sm'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'tc',tc_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tc'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sc',sc_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: sc'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      status=nfmpi_get_vara_real_all(ncid,tm_varid,start,edge,tm)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sm_varid,start,edge,sm)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,tc_varid,start,edge,tc)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sc_varid,start,edge,sc)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end


!_______________________________________________________________________
      subroutine read_ssts_monthly_pnetcdf(sst,sss,in_file)
! read wind stress
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer n
      real sst(im,jm),sss(im,jm)
      integer i,j
      character*9 in_file
      character*120 netcdf_ssts_file
      integer ncid,status
      integer sst_varid,sss_varid
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)

! open netcdf wind file
      write(netcdf_ssts_file,'(''in/ssts/'',a)') trim(in_file)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                            trim(netcdf_ssts_file)
      status=nfmpi_open(pom_comm,netcdf_ssts_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_ssts_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'sst',sst_varid)
!     status=nfmpi_inq_varid(ncid,'SST',sst_varid) !lyo:pac10:AluUpperSST
      call handle_error_pnetcdf('nfmpi_inq_varid: sst'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'sss',sss_varid)
!     status=nfmpi_inq_varid(ncid,'SSS',sss_varid) !lyo:pac10:AluUpperSSS
      call handle_error_pnetcdf('nfmpi_inq_varid: sss'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_get_vara_real_all(ncid,sst_varid,start,edge,sst)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(ncid,sss_varid,start,edge,sss)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________

      subroutine read_uabe_pnetcdf(uabe_out) !lyo:pac10:use stcc version
! read transport at the eastern boundary for PROFS in GOM !ayumi 2010/4/7
!lyo:20110224:alu:stcc:modified to work if *bc.nc* is not provided
!     as in STCC idealized run
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character*120 netcdf_bc_file
      integer ncid,status
      integer uabe_varid
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)
      real uabe_out(jm)
      logical :: fexist !lyo:20110224:alu:stcc:
!
! open netcdf ic file
      write(netcdf_bc_file,'(a)') "./in/bc.nc"

! check if file *bc* exist or not !lyo:20110224:alu:stcc:
      inquire(file=trim(netcdf_bc_file),exist=fexist)
!
      if(fexist) then !yes, exist, continue to read...
!
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_bc_file)
      status=nfmpi_open(pom_comm,netcdf_bc_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_bc_file,
     $                          status,nf_noerr)
! get variables
      status=nfmpi_inq_varid(ncid,'UABE',uabe_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uabe',status,nf_noerr)
! get data
      start(1)=j_global(1)
      edge(1)=jm
      status=nfmpi_get_vara_real_all(
     $ ncid,uabe_varid,start,edge,uabe_out)
      call handle_error_pnetcdf(
     $ 'nfmpi_get_vara_real_all: uabe',status,nf_noerr)
! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)
!
      else !lyo:no, does not exist, give warning...

      if(my_task.eq.master_task) then
!----------------------------------------------------------------------!
         write(6,'(/''****Warning from subr. read_uabe_pnetcdf****'')')
         write(6,'(/''uabe file '',a)')trim(netcdf_bc_file)
         write(6,'(/''does not exist, and uabe set to zero'')')
         uabe_out(:)=0.0
      end if

      endif

      return
      end
!_______________________________________________________________________

      subroutine read_tide_east_pnetcdf
     $           (ampe_out,phae_out,amue_out,phue_out)
!fhx:tide:read tide at the eastern boundary for PROFS
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character*120 netcdf_bc_file
      integer ncid,status
      integer ampe_varid,phae_varid,amue_varid,phue_varid
      integer vdims(2)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(2),edge(2)
      real ampe_out(jm,ntide),phae_out(jm,ntide)
      real amue_out(jm,ntide),phue_out(jm,ntide)

! open netcdf ic file
      write(netcdf_bc_file,'(a)') "./in/tide.nc"

      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $  trim(netcdf_bc_file)
      status=nfmpi_open(pom_comm,netcdf_bc_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_bc_file,
     $                          status,nf_noerr)


! get variables
      status=nfmpi_inq_varid(ncid,'ampe',ampe_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ampe',status,nf_noerr)

      status=nfmpi_inq_varid(ncid,'phae',phae_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: phae',status,nf_noerr)

      status=nfmpi_inq_varid(ncid,'amue',amue_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: amue',status,nf_noerr)

      status=nfmpi_inq_varid(ncid,'phue',phue_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: phue',status,nf_noerr)
      

! get data
      start(1)=j_global(1)
      start(2)=1
      edge(1)=jm
      edge(2)=ntide
      status=nfmpi_get_vara_real_all(
     $ ncid,ampe_varid,start,edge,ampe_out)
      call handle_error_pnetcdf(
     $ 'nfmpi_get_vara_real_all: ampe',status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $ ncid,phae_varid,start,edge,phae_out)
      call handle_error_pnetcdf(
     $ 'nfmpi_get_vara_real_all: phae',status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $ ncid,amue_varid,start,edge,amue_out)
      call handle_error_pnetcdf(
     $ 'nfmpi_get_vara_real_all: amue',status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $ ncid,phue_varid,start,edge,phue_out)
      call handle_error_pnetcdf(
     $ 'nfmpi_get_vara_real_all: phue',status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________

!_______________________________________________________________________
      subroutine read_wind_pnetcdf( uwnd_out, vwnd_out, filename ) !lyo:pac10:use stcc version !lyo:pac10:not used!
! read wind stress
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer n
      integer, parameter :: wind_npd = 4
      real uwnd_out(im,jm,wind_npd),vwnd_out(im,jm,wind_npd)
      integer i,j
      character :: filename*16, netcdf_wind_file*24
      integer ncid,status
      integer uwnd_varid,vwnd_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf wind file
      netcdf_wind_file="in/"//windf//"/"//trim(filename)
!*    write(netcdf_wind_file,'("in/"//windf//"/",a )')
!     write(netcdf_wind_file,'("in/"//trim(windf)//"/",a )')
!!    write(netcdf_wind_file,'("in/"trim(windf)"/",a )')
!*   $                         trim(filename)
!!    write(netcdf_wind_file,'(''in/gfsw/'',a )') trim(filename)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_wind_file)
      status=nfmpi_open(pom_comm,netcdf_wind_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_wind_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'uwnd',uwnd_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uwnd'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vwnd',vwnd_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vwnd'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=wind_npd

      status=nfmpi_get_vara_real_all(
     $     ncid,uwnd_varid,start,edge,uwnd_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(
     $     ncid,vwnd_varid,start,edge,vwnd_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________
!_______________________________________________________________________
      subroutine read_wind_pnetcdfc( uwnd_out, vwnd_out, filename )
! read wind stress
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      integer n
      integer, parameter :: wind_npd = 4
      real uwnd_out(im_coarse,jm_coarse,wind_npd)
      real vwnd_out(im_coarse,jm_coarse,wind_npd)
      integer i,j
!      character :: filename*15, netcdf_wind_file*23
      character :: filename*16, netcdf_wind_file*24
      integer ncid,status
      integer uwnd_varid,vwnd_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf wind file
      netcdf_wind_file="in/"//windf//"/"//trim(filename)
!*    write(netcdf_wind_file,'("in/"//windf//"/",a )')
!     write(netcdf_wind_file,'("in/"//trim(windf)//"/",a )')
!!    write(netcdf_wind_file,'("in/"trim(windf)"/",a )')
!*   $                         trim(filename)
!!    write(netcdf_wind_file,'(''in/gfsw/'',a )') trim(filename)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_wind_file)
      status=nfmpi_open(pom_comm_coarse,netcdf_wind_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_wind_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'uwnd',uwnd_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uwnd'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vwnd',vwnd_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vwnd'
     $                          ,status,nf_noerr)

! get data
      start(1)=i_global_coarse(1)
      start(2)=j_global_coarse(1)
      start(3)=1
      edge(1)=im_coarse
      edge(2)=jm_coarse
      edge(3)=wind_npd

      status=nfmpi_get_vara_real_all(
     $     ncid,uwnd_varid,start,edge,uwnd_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)
      status=nfmpi_get_vara_real_all(
     $     ncid,vwnd_varid,start,edge,vwnd_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________

!_______________________________________________________________________
      subroutine read_msla_pnetcdf( ssha_out, filename )
! read mean sea level anomaly data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      real ssha_out(im,jm)    
      real errval
      character :: filename*16, netcdf_msla_file*25
      integer ncid,status
      integer ssha_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf msla file
      write(netcdf_msla_file,'(''in/assim/'',a )') trim(filename)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_msla_file)
      status=nfmpi_open(pom_comm,netcdf_msla_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_msla_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'ssha',ssha_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ssha'
     $                          ,status,nf_noerr)
      status=nfmpi_get_att_real(ncid, ssha_varid,
     $     'missing_value', errval )
!      if(my_task.eq.0) write(*,'(a,f12.3)')
!     $                                  " missing_value = ", errval
      

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=1

      status=nfmpi_get_vara_real_all(
     $     ncid,ssha_varid,start,edge,ssha_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)


! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________
!_______________________________________________________________________
      subroutine read_msla_pnetcdfc( ssha_out, filename )
! read mean sea level anomaly data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      real ssha_out(im_coarse,jm_coarse)    
      real errval
      character :: filename*16, netcdf_msla_file*25
      integer ncid,status
      integer ssha_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf msla file
      write(netcdf_msla_file,'(''in/assim/'',a )') trim(filename)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_msla_file)
      status=nfmpi_open(pom_comm_coarse,netcdf_msla_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_msla_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'ssha',ssha_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ssha'
     $                          ,status,nf_noerr)
      status=nfmpi_get_att_real(ncid, ssha_varid,
     $     'missing_value', errval )
!      if(my_task.eq.0) write(*,'(a,f12.3)')
!     $                                  " missing_value = ", errval
      

! get data
      start(1)=i_global_coarse(1)
      start(2)=j_global_coarse(1)
      start(3)=1
      edge(1)=im_coarse
      edge(2)=jm_coarse
      edge(3)=1

      status=nfmpi_get_vara_real_all(
     $     ncid,ssha_varid,start,edge,ssha_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)


! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________

!_______________________________________________________________________
      subroutine read_assiminfo_pnetcdf
     $     ( frs_out, tav_out, fac_out, cof_out )
! read information for data assimilation
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      real, dimension(im_local,jm_local) :: frs_out
      real, dimension(im_local,jm_local,kb) 
     $     :: tav_out, fac_out, cof_out
      character :: netcdf_assiminfo_file*21
      integer ncid,status
      integer frs_varid, tav_varid, fac_varid, cof_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf assiminfo file
      write(netcdf_assiminfo_file,'(''in/assim/'',a )') "assiminfo.nc"
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_assiminfo_file)
      status=nfmpi_open(pom_comm,netcdf_assiminfo_file,nf_nowrite,
     $                  mpi_info_null,ncid)

! get variables
      status=nfmpi_inq_varid(ncid,'frs',frs_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: frs'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'tav',tav_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tav'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'fac',fac_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: fac'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cof',cof_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cof'
     $                          ,status,nf_noerr)
  


! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=1

      status=nfmpi_get_vara_real_all(
     $     ncid,frs_varid,start,edge,frs_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb

      status=nfmpi_get_vara_real_all(
     $     ncid,tav_varid,start,edge,tav_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $     ncid,fac_varid,start,edge,fac_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $     ncid,cof_varid,start,edge,cof_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)


! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________
!_______________________________________________________________________
      subroutine read_assiminfo_pnetcdfc
     $     ( frs_out, tav_out, fac_out, cof_out )
! read information for data assimilation
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      real, dimension(im_local_coarse,jm_local_coarse) :: frs_out
      real, dimension(im_local_coarse,jm_local_coarse,kb) 
     $     :: tav_out, fac_out, cof_out
      character :: netcdf_assiminfo_file*21
      integer ncid,status
      integer frs_varid, tav_varid, fac_varid, cof_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf assiminfo file
      write(netcdf_assiminfo_file,'(''in/assim/'',a )') "assiminfo.nc"
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_assiminfo_file)
      status=nfmpi_open(pom_comm_coarse,netcdf_assiminfo_file,
     $                 nf_nowrite,mpi_info_null,ncid)

! get variables
      status=nfmpi_inq_varid(ncid,'frs',frs_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: frs'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'tav',tav_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: tav'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'fac',fac_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: fac'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cof',cof_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cof'
     $                          ,status,nf_noerr)
  


! get data
      start(1)=i_global_coarse(1)
      start(2)=j_global_coarse(1)
      start(3)=1
      edge(1)=im_coarse
      edge(2)=jm_coarse
      edge(3)=1

      status=nfmpi_get_vara_real_all(
     $     ncid,frs_varid,start,edge,frs_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      start(1)=i_global_coarse(1)
      start(2)=j_global_coarse(1)
      start(3)=1
      edge(1)=im_coarse
      edge(2)=jm_coarse
      edge(3)=kb

      status=nfmpi_get_vara_real_all(
     $     ncid,tav_varid,start,edge,tav_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $     ncid,fac_varid,start,edge,fac_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)

      status=nfmpi_get_vara_real_all(
     $     ncid,cof_varid,start,edge,cof_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)


! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!_______________________________________________________________________
!_______________________________________________________________________
!fhx:mcsst:start
      subroutine read_mcsst_pnetcdf( mcsst_out, filename )
! read mean MCSST data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      real mcsst_out(im_local,jm_local)    
      real errval
      character :: filename*17, netcdf_mcsst_file*26
      integer ncid,status
      integer mcsst_varid
      integer vdims(3)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(3),edge(3)


! open netcdf mcsst file
      write(netcdf_mcsst_file,'(''in/mcsst/'',a )') trim(filename)
      if(my_task.eq.0) write(*,'(/''reading file '',a)')
     $                                    trim(netcdf_mcsst_file)
      status=nfmpi_open(pom_comm,netcdf_mcsst_file,nf_nowrite,
     $                  mpi_info_null,ncid)
      call handle_error_pnetcdf('nfmpi_open: '//netcdf_mcsst_file,
     $                          status,nf_noerr)

! get variables
      status=nfmpi_inq_varid(ncid,'mcsst',mcsst_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: mcsst'
     $                          ,status,nf_noerr)
      status=nfmpi_get_att_real(ncid, mcsst_varid,
     $     'missing_value', errval )
!      if(my_task.eq.0) write(*,'(a,f12.3)')
!     $                                  " missing_value = ", errval
      

! get data
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=1

      status=nfmpi_get_vara_real_all(
     $     ncid,mcsst_varid,start,edge,mcsst_out)
      call handle_error_pnetcdf('nfmpi_get_vara_real_all',
     $                          status,nf_noerr)


! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close',status,nf_noerr)

      return
      end
!fhx:mcsst:end
!_______________________________________________________________________
!_______________________________________________________________________

      subroutine write_output_pnetcdf( netcdf_out_file ) !lyo:pac10:use stcc version
! write output data
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer time_dimid,x_dimid,y_dimid,z_dimid,nb_dimid !fhx:tracer!lyo:tracer
      integer z_varid,zz_varid,dx_varid,dy_varid,east_c_varid,
     $        east_e_varid,east_u_varid,east_v_varid,north_c_varid,
     $        north_e_varid,north_u_varid,north_v_varid,rot_varid,
     $        h_varid,fsm_varid,dum_varid,dvm_varid,
     $        wusurf_varid,wvsurf_varid,wtsurf_varid,wssurf_varid
      integer time_varid,uab_varid,vab_varid,elb_varid,u_varid,v_varid,
     $        w_varid,t_varid,s_varid,rho_varid,km_varid,kh_varid
      integer tr_varid  !lyo:tracer
      integer ncid,status
      integer vdims(5)  !lyo:tracer
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(5),edge(5)  !lyo:tracer
      
      iout = iout + 1

      if ( iout == 1  ) then

!     create netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
         status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
         call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=nf_unlimited
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)
!fhx:tracer:begin
      length=nb
      status=nfmpi_def_dim(ncid,'nb',length,nb_dimid)
      call handle_error_pnetcdf('nf_def_dim: nb',status,nf_noerr)
!fhx:tracer:end

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid
      call def_var_pnetcdf(ncid,'dx',2,vdims,dx_varid,
     $                     'grid increment in x','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'dy',2,vdims,dy_varid,
     $                     'grid increment in y','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'east_u',2,vdims,east_u_varid,
     $                     'easting of u-points','degree',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'east_v',2,vdims,east_v_varid,
     $                     'easting of v-points','degree',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'east_e',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'east_c',2,vdims,east_c_varid,
     $                     'easting of cell corners','degree',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'north_u',2,vdims,north_u_varid,
     $                     'northing of u-points','degree',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'north_v',2,vdims,north_v_varid,
     $                     'northing of v-points','degree',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'north_e',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'north_c',2,vdims,north_c_varid,
     $                     'northing of cell corners','degree',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'rot',2,vdims,rot_varid,
     $                    'Rotation angle of x-axis wrt. east','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'dum',2,vdims,dum_varid,
     $                     'u-velocity mask','dimensionless',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'dvm',2,vdims,dvm_varid,
     $                     'v-velocity mask','dimensionless',
     $                     'east_v north_v',.true.)
     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'uab',3,vdims,uab_varid,
     $                     'depth-averaged u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vab',3,vdims,vab_varid,
     $                     'depth-averaged v','metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'elb',3,vdims,elb_varid,
     $                     'surface elevation','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'wusurf',3,vdims,wusurf_varid,
     $                     'x-momentum flux','metre^2/sec^2',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'wvsurf',3,vdims,wvsurf_varid,
     $                     'y-momentum flux','metre^2/sec^2',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'wtsurf',3,vdims,wtsurf_varid,
     $                     'temperature flux','deg m/s',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'wssurf',3,vdims,wssurf_varid,
     $                     'salinity flux','psu m/s',
     $                     'east_e north_e',.true.)
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=time_dimid
      call def_var_pnetcdf(ncid,'u',4,vdims,u_varid,
     $                     'x-velocity','metre/sec',
     $                     'east_u north_u zz',.true.)
      call def_var_pnetcdf(ncid,'v',4,vdims,v_varid,
     $                     'y-velocity','metre/sec',
     $                     'east_v north_v zz',.true.)
      call def_var_pnetcdf(ncid,'w',4,vdims,w_varid,
     $                     'z-velocity','metre/sec',
     $                     'east_e north_e z',.true.)
      call def_var_pnetcdf(ncid,'t',4,vdims,t_varid,
     $                     'potential temperature','K',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'s',4,vdims,s_varid,
     $                     'salinity x rho / rhoref','PSS',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'rho',4,vdims,rho_varid,
     $                     '(density-1000)/rhoref','dimensionless',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'kh',4,vdims,kh_varid,
     $                     'vertical diffusivity','metre2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'km',4,vdims,km_varid,
     $                     'vertical viscosity','metre2/sec',
     $                     'east_e north_e zz',.true.)

!lyo:tracer:begin
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=nb_dimid
      vdims(5)=time_dimid
      call def_var_pnetcdf(ncid,'tr',5,vdims,tr_varid,
     $                     'tracer ','dimensionless',
     $                     'east_u north_u zz nb',.true.)
!lyo:tracer:end
! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
!      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_put_vara_real_all(ncid,dx_varid,start,edge,dx)
      call handle_error_pnetcdf('nf_put_var_real: dx',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dy_varid,start,edge,dy)
      call handle_error_pnetcdf('nf_put_var_real: dy',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_u_varid,start,edge,
     $                               east_u)
      call handle_error_pnetcdf('nf_put_var_real: east_u',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_v_varid,start,edge,
     $                               east_v)
      call handle_error_pnetcdf('nf_put_var_real: east_v',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_c_varid,start,edge,
     $                               east_c)
      call handle_error_pnetcdf('nf_put_var_real: east_c',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_u_varid,start,edge,
     $                               north_u)
      call handle_error_pnetcdf('nf_put_var_real: north_u',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_v_varid,start,edge,
     $                               north_v)
      call handle_error_pnetcdf('nf_put_var_real: north_v',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_c_varid,start,edge,
     $                               north_c)
      call handle_error_pnetcdf('nf_put_var_real: north_c',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rot_varid,start,edge,rot)
      call handle_error_pnetcdf('nf_put_var_real: rot',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dum_varid,start,edge,dum)
      call handle_error_pnetcdf('nf_put_var_real: dum',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dvm_varid,start,edge,dvm)
      call handle_error_pnetcdf('nf_put_var_real: dvm',status,nf_noerr)


      else


!     open netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
      status=nfmpi_open(pom_comm,trim(netcdf_out_file),nf_write,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_open: '//netcdf_out_file,
     $                          status,nf_noerr)


!!     inquire dimensions
!
!      status=nfmpi_inq_dimid(ncid,'time',time_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: time',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'z',z_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: z',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'y',y_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: y',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'x',x_dimid)
!     call handle_error_pnetcdf('nfmpi_inq_dim: x',status,nf_noerr)



!     inqure variables
      status=nfmpi_inq_varid(ncid,'time',time_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: time'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'uab',uab_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uab'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vab',vab_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vab'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'elb',elb_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: elb'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wusurf',wusurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wusurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wvsurf',wvsurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wvsurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wtsurf',wtsurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wtsurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'wssurf',wssurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: wssurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'u',u_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: u'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'v',v_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: v'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'w',w_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: w'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'t',t_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: t'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'s',s_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: s'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'rho',rho_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: rho'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'kh',kh_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: kh'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'km',km_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: km'
     $                          ,status,nf_noerr)
!lyo:tracer:begin
      status=nfmpi_inq_varid(ncid,'tr',tr_varid)
!     call handle_error_pnetcdf('nfmpi_inq_varid: tr_mean' !should be tr_mean?
      call handle_error_pnetcdf('nfmpi_inq_varid: tr'
     $                          ,status,nf_noerr)
!lyo:tracer:end

      endif !if ( iout == 1  ) then...

!     write data

      if ( iout == 1  ) then !lyo:20110224:stcc:use orig t, not t_mean

      start(1)=iout
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=iout
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(ncid,uab_varid,start,edge,uab)
      call handle_error_pnetcdf('nf_put_vara_real: uab',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,vab_varid,start,edge,vab)
      call handle_error_pnetcdf('nf_put_vara_real: vab',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,elb_varid,start,edge,elb)
      call handle_error_pnetcdf('nf_put_vara_real: elb',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wusurf_varid,start,edge,wusurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wusurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wvsurf_varid,start,edge,wvsurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wvsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wtsurf_varid,start,edge,wtsurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wtsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wssurf_varid,start,edge,wssurf)
      call handle_error_pnetcdf('
     $     nf_put_vara_real: wssurf',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=iout
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=1
      status=nfmpi_put_vara_real_all(ncid,u_varid,start,edge,u)
      call handle_error_pnetcdf('nf_put_vara_real: u',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,v_varid,start,edge,v)
      call handle_error_pnetcdf('nf_put_vara_real: v',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,w_varid,start,edge,w)
      call handle_error_pnetcdf('nf_put_vara_real: w',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,t_varid,start,edge,t)
      call handle_error_pnetcdf('nf_put_vara_real: t',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,s_varid,start,edge,s)
      call handle_error_pnetcdf('nf_put_vara_real: s',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rho_varid,start,edge,rho)
      call handle_error_pnetcdf('nf_put_vara_real: rho',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,kh_varid,start,edge,kh)
      call handle_error_pnetcdf('nf_put_vara_real: kh',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,km_varid,start,edge,km)
      call handle_error_pnetcdf('nf_put_vara_real: km',status,nf_noerr)

!lyo:tracer:begin
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      start(5)=iout
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=nb
      edge(5)=1
!     status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr_mean)
      status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr)
      call handle_error_pnetcdf('nf_put_vara_real: tr',status,nf_noerr)
!lyo:tracer:end
! close file

      else !lyo:20110224:stcc:save *_mean

      start(1)=iout
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=iout
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(ncid,uab_varid,start,edge,uab_mean)
      call handle_error_pnetcdf('nf_put_vara_real: uab',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,vab_varid,start,edge,vab_mean)
      call handle_error_pnetcdf('nf_put_vara_real: vab',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,elb_varid,start,edge,elb_mean)
      call handle_error_pnetcdf('nf_put_vara_real: elb',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wusurf_varid,start,edge,wusurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wusurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wvsurf_varid,start,edge,wvsurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wvsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wtsurf_varid,start,edge,wtsurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wtsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wssurf_varid,start,edge,wssurf_mean)
      call handle_error_pnetcdf('
     $     nf_put_vara_real: wssurf',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=iout
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=1
      status=nfmpi_put_vara_real_all(ncid,u_varid,start,edge,u_mean)
      call handle_error_pnetcdf('nf_put_vara_real: u',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,v_varid,start,edge,v_mean)
      call handle_error_pnetcdf('nf_put_vara_real: v',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,w_varid,start,edge,w_mean)
      call handle_error_pnetcdf('nf_put_vara_real: w',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,t_varid,start,edge,t_mean)
      call handle_error_pnetcdf('nf_put_vara_real: t',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,s_varid,start,edge,s_mean)
      call handle_error_pnetcdf('nf_put_vara_real: s',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rho_varid,start,edge,rho_mean)
      call handle_error_pnetcdf('nf_put_vara_real: rho',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,kh_varid,start,edge,kh_mean)
      call handle_error_pnetcdf('nf_put_vara_real: kh',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,km_varid,start,edge,km_mean)
      call handle_error_pnetcdf('nf_put_vara_real: km',status,nf_noerr)

!lyo:tracer:begin
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      start(5)=iout
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=nb
      edge(5)=1
!     status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr_mean)
      status=nfmpi_put_vara_real_all(ncid,tr_varid,start,edge,tr)
      call handle_error_pnetcdf('nf_put_vara_real: tr',status,nf_noerr)
!lyo:tracer:end

      endif !if ( iout == 1  ) then... !lyo:20110224:stcc:

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________

!_______________________________________________________________________
      subroutine write_SURF_pnetcdf( netcdf_out_file )!fhx:20110131:new subr
!     Version with writing u,vtf_mean
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer time_dimid,x_dimid,y_dimid,z_dimid
      integer z_varid,zz_varid,
     $        east_e_varid,
     $        north_e_varid,
     $        h_varid,fsm_varid
      integer time_varid
      integer usrf_varid,vsrf_varid,elsrf_varid,uwsrf_varid,vwsrf_varid
     $       ,utf_varid,vtf_varid !lyo:vort:
      integer ncid,status
      integer vdims(4)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4)
      

      iouts = iouts + 1

      if ( iouts == 1  ) then

!     create netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
         status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
         call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=nf_unlimited
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid

      call def_var_pnetcdf(ncid,'alon',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'alat',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)

      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)

     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'usrf',3,vdims,usrf_varid,
     $                     'surface u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vsrf',3,vdims,vsrf_varid,
     $                     'surface v','metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'elsrf',3,vdims,elsrf_varid,
     $                     'surface elevation','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'uwsrf',3,vdims,uwsrf_varid,
     $                     'wind u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vwsrf',3,vdims,vwsrf_varid,
     $                     'wind v','metre/sec',
     $                     'east_v north_v',.true.)
!lyo:vort:
      call def_var_pnetcdf(ncid,'utf',3,vdims,utf_varid,
     $                     'u-transport','m^2/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vtf',3,vdims,vtf_varid,
     $                     'v-transport','m^2/sec',
     $                     'east_v north_v',.true.)
! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
!      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm

      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)

      else

!     open netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
      status=nfmpi_open(pom_comm,trim(netcdf_out_file),nf_write,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_open: '//netcdf_out_file,
     $                          status,nf_noerr)

!!     inquire dimensions
!
!      status=nfmpi_inq_dimid(ncid,'time',time_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: time',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'z',z_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: z',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'y',y_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: y',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'x',x_dimid)
!     call handle_error_pnetcdf('nfmpi_inq_dim: x',status,nf_noerr)



!     inqure variables
      status=nfmpi_inq_varid(ncid,'time',time_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: time'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'usrf',usrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: usrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vsrf',vsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'elsrf',elsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: elsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'uwsrf',uwsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uwsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vwsrf',vwsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vwsrf'
     $                          ,status,nf_noerr)
!lyo:vort:
      status=nfmpi_inq_varid(ncid,'utf',utf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: utf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vtf',vtf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vtf'
     $                          ,status,nf_noerr)

      endif

!     write data

      start(1)=iouts
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=iouts
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(
     $      ncid,usrf_varid,start,edge,usrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: usrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vsrf_varid,start,edge,vsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,elsrf_varid,start,edge,elsrf_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: elsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,uwsrf_varid,start,edge,uwsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: uwsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vwsrf_varid,start,edge,vwsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vwsrf',status,nf_noerr)
!lyo:vort:
      status=nfmpi_put_vara_real_all(
     $      ncid,utf_varid,start,edge,utf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: utf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vtf_varid,start,edge,vtf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vtf',status,nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________
!_______________________________________________________________________
      subroutine write_SURF_pnetcdf_ori( netcdf_out_file )!fhx:20110131:new subr
!     Original version without writing u,vtf_mean
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer time_dimid,x_dimid,y_dimid,z_dimid
      integer z_varid,zz_varid,
     $        east_e_varid,
     $        north_e_varid,
     $        h_varid,fsm_varid
      integer time_varid
      integer usrf_varid,vsrf_varid,elsrf_varid,uwsrf_varid,vwsrf_varid
      integer ncid,status
      integer vdims(4)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4)
      

      iouts = iouts + 1

      if ( iouts == 1  ) then

!     create netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
         status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
         call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=nf_unlimited
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid

      call def_var_pnetcdf(ncid,'alon',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'alat',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)

      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)

     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'usrf',3,vdims,usrf_varid,
     $                     'surface u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vsrf',3,vdims,vsrf_varid,
     $                     'surface v','metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'elsrf',3,vdims,elsrf_varid,
     $                     'surface elevation','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'uwsrf',3,vdims,uwsrf_varid,
     $                     'wind u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vwsrf',3,vdims,vwsrf_varid,
     $                     'wind v','metre/sec',
     $                     'east_v north_v',.true.)


! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
!      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm

      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)

      else

!     open netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
      status=nfmpi_open(pom_comm,trim(netcdf_out_file),nf_write,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_open: '//netcdf_out_file,
     $                          status,nf_noerr)

!!     inquire dimensions
!
!      status=nfmpi_inq_dimid(ncid,'time',time_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: time',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'z',z_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: z',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'y',y_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: y',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'x',x_dimid)
!     call handle_error_pnetcdf('nfmpi_inq_dim: x',status,nf_noerr)



!     inqure variables
      status=nfmpi_inq_varid(ncid,'time',time_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: time'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'usrf',usrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: usrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vsrf',vsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'elsrf',elsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: elsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'uwsrf',uwsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uwsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vwsrf',vwsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vwsrf'
     $                          ,status,nf_noerr)

      endif

!     write data

      start(1)=iouts
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=iouts
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(
     $      ncid,usrf_varid,start,edge,usrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: usrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vsrf_varid,start,edge,vsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,elsrf_varid,start,edge,elsrf_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: elsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,uwsrf_varid,start,edge,uwsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: uwsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vwsrf_varid,start,edge,vwsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vwsrf',status,nf_noerr)
 

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________
!lyo:!stokes:beg:
!_______________________________________________________________________
      subroutine write_SURFStokes_pnetcdf( netcdf_out_file )
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer time_dimid,x_dimid,y_dimid,z_dimid
      integer z_varid,zz_varid,
     $        east_e_varid,
     $        north_e_varid,
     $        h_varid,fsm_varid
      integer time_varid
      integer usrf_varid,vsrf_varid,elsrf_varid,uwsrf_varid,vwsrf_varid,
     $        xstks_varid,ystks_varid
      integer ncid,status
      integer vdims(4)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4)
      integer i,j,k
      

      iouts = iouts + 1

      if ( iouts == 1  ) then

!     create netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
         status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
         call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=nf_unlimited
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid

      call def_var_pnetcdf(ncid,'alon',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'alat',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)

      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)

     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'usrf',3,vdims,usrf_varid,
     $                     'surface u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vsrf',3,vdims,vsrf_varid,
     $                     'surface v','metre/sec',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'elsrf',3,vdims,elsrf_varid,
     $                     'surface elevation','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'uwsrf',3,vdims,uwsrf_varid,
     $                     'wind u','metre/sec',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'vwsrf',3,vdims,vwsrf_varid,
     $                     'wind v','metre/sec',
     $                     'east_v north_v',.true.)
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=time_dimid
      call def_var_pnetcdf(ncid,'xstks',4,vdims,xstks_varid,
     $                     'x-Stks Acc & Vel','m/s2 & m/s',
     $                     'east_u north_u zz',.true.)
      call def_var_pnetcdf(ncid,'ystks',4,vdims,ystks_varid,
     $                     'y-Stks Acc & Vel','m/s2 & m/s',
     $                     'east_v north_v zz',.true.)

! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
!      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm

      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)

      else

!     open netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
      status=nfmpi_open(pom_comm,trim(netcdf_out_file),nf_write,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_open: '//netcdf_out_file,
     $                          status,nf_noerr)

!!     inquire dimensions
!
!      status=nfmpi_inq_dimid(ncid,'time',time_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: time',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'z',z_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: z',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'y',y_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: y',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'x',x_dimid)
!     call handle_error_pnetcdf('nfmpi_inq_dim: x',status,nf_noerr)



!     inqure variables
      status=nfmpi_inq_varid(ncid,'time',time_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: time'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'usrf',usrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: usrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vsrf',vsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'elsrf',elsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: elsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'uwsrf',uwsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: uwsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'vwsrf',vwsrf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: vwsrf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'xstks',xstks_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: xstks'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ystks',ystks_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ystks'
     $                          ,status,nf_noerr)

      endif

!     write data

      start(1)=iouts
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=iouts
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(
     $      ncid,usrf_varid,start,edge,usrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: usrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vsrf_varid,start,edge,vsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,elsrf_varid,start,edge,elsrf_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: elsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,uwsrf_varid,start,edge,uwsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: uwsrf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,vwsrf_varid,start,edge,vwsrf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: vwsrf',status,nf_noerr)
 
      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=iouts
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=1
! put Stokes forces in 1:kb/2, vel in kb/2+1:kb
!     Keep k=kb to be bottom-most Stokes forces
      xstks_mean(:,:,kb)=xstks_mean(:,:,kbm1)
      ystks_mean(:,:,kb)=ystks_mean(:,:,kbm1)
      i=int(kb/2)
      do k=i+1,kbm1
         xstks_mean(:,:,k)=ustks(:,:,k-i)
         ystks_mean(:,:,k)=vstks(:,:,k-i)
         enddo
      status=nfmpi_put_vara_real_all(ncid,xstks_varid,start,edge,
     $     xstks_mean)
      call handle_error_pnetcdf('nf_put_vara_real: xstks',status,
     $     nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,ystks_varid,start,edge,
     $     ystks_mean)
      call handle_error_pnetcdf('nf_put_vara_real: ystks',status,
     $     nf_noerr)

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________
!lyo:!stokes:end:
!_______________________________________________________________________
      subroutine write_output_init_pnetcdf( netcdf_out_file ) !lyo:20110224:alu:stcc:!lyo:pac10:add this subroutine
!alu:20100614:  write output data for initial fields   
!alu:20110105: change the name and clean up a little bit
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp !,netcdf_out_file !lyo:20110224:alu:stcc:
      integer nprint
      integer time_dimid,x_dimid,y_dimid,z_dimid
      integer time_varid
     $       ,z_varid,zz_varid,dx_varid,dy_varid
     $       , east_c_varid, east_e_varid, east_u_varid, east_v_varid
     $       ,north_c_varid,north_e_varid,north_u_varid,north_v_varid
     $       ,h_varid,fsm_varid,dum_varid,dvm_varid,rot_varid
     $       ,wusurf_varid,wvsurf_varid,wtsurf_varid,wssurf_varid
     $       ,t_varid,s_varid,rho_varid,km_varid,kh_varid
     $       ,frz_varid
     $       ,aamfrz_varid
      integer ele_varid,elw_varid,els_varid,eln_varid
      integer ncid,status,flags
      integer vdims(4)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4)
      

!     netcdf_out_file='./out/out.init.nc' !lyo:20110224:alu:stcc:
      if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $     trim(netcdf_out_file)

      flags= NF_CLOBBER + NF_64BIT_OFFSET  !for large file( > 2GB) !lyo:pac10:
!     flags= NF_CLOBBER                     !for small file( < 2GB)

      status=nfmpi_create(pom_comm,trim(netcdf_out_file),flags,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=1
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'sigma',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      str_tmp='ocean_sigma_coordinate'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,str_tmp)
      str_tmp='sigma: z eta: elb depth: h'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,str_tmp)
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      str_tmp='ocean_sigma_coordinate'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,str_tmp)
      str_tmp='sigma: zz eta: elb depth: h'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,str_tmp)
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

!alu:add calculated elebc output
      vdims(1)=x_dimid 
      call def_var_pnetcdf(ncid,'els',1,vdims,els_varid, !South
     $                      'South elebc','metre',
     $                      'east_e',.true.)
      call def_var_pnetcdf(ncid,'eln',1,vdims,eln_varid, !North
     $                      'Nouth elebc','metre',
     $                      'east_e',.true.)
      vdims(1)=y_dimid
      call def_var_pnetcdf(ncid,'ele',1,vdims,ele_varid, !East
     $                      'East elebc','metre',
     $                      'north_e',.true.)
      call def_var_pnetcdf(ncid,'elw',1,vdims,elw_varid, !West
     $                      'West elebc','metre',
     $                      'north_e',.true.)


      vdims(1)=x_dimid
      vdims(2)=y_dimid
      call def_var_pnetcdf(ncid,'dx',2,vdims,dx_varid,
     $                     'grid increment in x','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'dy',2,vdims,dy_varid,
     $                     'grid increment in y','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'east_u',2,vdims,east_u_varid,
     $                     'easting of u-points','degree',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'east_v',2,vdims,east_v_varid,
     $                     'easting of v-points','degree',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'east_e',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'east_c',2,vdims,east_c_varid,
     $                     'easting of cell corners','degree',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'north_u',2,vdims,north_u_varid,
     $                     'northing of u-points','degree',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'north_v',2,vdims,north_v_varid,
     $                     'northing of v-points','degree',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'north_e',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'north_c',2,vdims,north_c_varid,
     $                     'northing of cell corners','degree',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'rot',2,vdims,rot_varid,
     $                    'Rotation angle of x-axis wrt. east','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'dum',2,vdims,dum_varid,
     $                     'u-velocity mask','dimensionless',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'dvm',2,vdims,dvm_varid,
     $                     'v-velocity mask','dimensionless',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'frz',2,vdims,frz_varid,
     $                     'frz coeffi','dimensionless',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'aamfrz',2,vdims,aamfrz_varid,
     $                     'aamfrz coeffi','dimensionless',
     $                     'east_v north_v',.true.)
     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'wusurf',3,vdims,wusurf_varid,
     $                     'x-momentum flux','metre^2/sec^2',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'wvsurf',3,vdims,wvsurf_varid,
     $                     'y-momentum flux','metre^2/sec^2',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'wtsurf',3,vdims,wtsurf_varid,
     $                     'temperature flux','deg m/s',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'wssurf',3,vdims,wssurf_varid,
     $                     'salinity flux','psu m/s',
     $                     'east_e north_e',.true.)

      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=z_dimid
      vdims(4)=time_dimid
      call def_var_pnetcdf(ncid,'t',4,vdims,t_varid,
     $                     'potential temperature','K',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'s',4,vdims,s_varid,
     $                     'salinity x rho / rhoref','PSS',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'rho',4,vdims,rho_varid,
     $                     '(density-1000)/rhoref','dimensionless',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'kh',4,vdims,kh_varid,
     $                     'vertical diffusivity','metre2/sec',
     $                     'east_e north_e zz',.true.)
      call def_var_pnetcdf(ncid,'km',4,vdims,km_varid,
     $                     'vertical viscosity','metre2/sec',
     $                     'east_e north_e zz',.true.)

! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)


! write data
      start(1)=1
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

!alu:add elebc
!--East
       start(1)=j_global(1)
       edge(1)=jm
       status=nfmpi_put_vara_real_all(ncid,ele_varid,start,edge,ele)
       call handle_error_pnetcdf('nf_put_var_real: ele',status,nf_noerr)
!--West
       start(1)=j_global(1)
       edge(1)=jm
       status=nfmpi_put_vara_real_all(ncid,elw_varid,start,edge,elw)
       call handle_error_pnetcdf('nf_put_var_real: elw',status,nf_noerr)
!--South
       start(1)=i_global(1)
       edge(1)=im
       status=nfmpi_put_vara_real_all(ncid,els_varid,start,edge,els)
       call handle_error_pnetcdf('nf_put_var_real: els',status,nf_noerr)
!--North
       start(1)=i_global(1)
       edge(1)=im
       status=nfmpi_put_vara_real_all(ncid,eln_varid,start,edge,eln)
       call handle_error_pnetcdf('nf_put_var_real: eln',status,nf_noerr)

      

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm
      status=nfmpi_put_vara_real_all(ncid,dx_varid,start,edge,dx)
      call handle_error_pnetcdf('nf_put_var_real: dx',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dy_varid,start,edge,dy)
      call handle_error_pnetcdf('nf_put_var_real: dy',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_u_varid,start,edge,
     $                               east_u)
      call handle_error_pnetcdf('nf_put_var_real: east_u',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_v_varid,start,edge,
     $                               east_v)
      call handle_error_pnetcdf('nf_put_var_real: east_v',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,east_c_varid,start,edge,
     $                               east_c)
      call handle_error_pnetcdf('nf_put_var_real: east_c',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_u_varid,start,edge,
     $                               north_u)
      call handle_error_pnetcdf('nf_put_var_real: north_u',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_v_varid,start,edge,
     $                               north_v)
      call handle_error_pnetcdf('nf_put_var_real: north_v',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_c_varid,start,edge,
     $                               north_c)
      call handle_error_pnetcdf('nf_put_var_real: north_c',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rot_varid,start,edge,rot)
      call handle_error_pnetcdf('nf_put_var_real: rot',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dum_varid,start,edge,dum)
      call handle_error_pnetcdf('nf_put_var_real: dum',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,dvm_varid,start,edge,dvm)
      call handle_error_pnetcdf('nf_put_var_real: dvm',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,frz_varid,start,edge,frz)
      call handle_error_pnetcdf('nf_put_var_real: frz',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,aamfrz_varid,start,
     $                                                  edge,aamfrz)
      call handle_error_pnetcdf('nf_put_var_real: aamfrz',
     $                                             status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(
     $     ncid,wusurf_varid,start,edge,wusurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wusurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wvsurf_varid,start,edge,wvsurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wvsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wtsurf_varid,start,edge,wtsurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: wtsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,wssurf_varid,start,edge,wssurf)
      call handle_error_pnetcdf('
     $     nf_put_vara_real: wssurf',status,nf_noerr)


      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=1
      start(4)=1
      edge(1)=im
      edge(2)=jm
      edge(3)=kb
      edge(4)=1
      status=nfmpi_put_vara_real_all(ncid,t_varid,start,edge,tb)
      call handle_error_pnetcdf('nf_put_vara_real: t',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,s_varid,start,edge,sb)
      call handle_error_pnetcdf('nf_put_vara_real: s',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,rho_varid,start,edge,rho)
      call handle_error_pnetcdf('nf_put_vara_real: rho',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,kh_varid,start,edge,kh)
      call handle_error_pnetcdf('nf_put_vara_real: kh',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,km_varid,start,edge,km)
      call handle_error_pnetcdf('nf_put_vara_real: km',status,nf_noerr)


! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output: '
     $                           ,status,nf_noerr)

      return
      end
!_______________________________________________________________________
!lyo:!vort:beg:Write vorticity(e.g. JEBAR) analysis
!_______________________________________________________________________
      subroutine write_vort_pnetcdf_not_mean( netcdf_out_file )
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer time_dimid,x_dimid,y_dimid,z_dimid
      integer z_varid,zz_varid,
     $        east_e_varid,
     $        north_e_varid,
     $        h_varid,fsm_varid
      integer time_varid
      integer totx_varid,toty_varid,celg_varid,ctsurf_varid
      integer ctbot_varid,cpvf_varid,cjbar_varid,cadv_varid,cten_varid
      integer ncid,status
      integer vdims(4)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4)
      

      ioutv = ioutv + 1

      if ( ioutv == 1  ) then

!     create netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
         status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
         call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=nf_unlimited
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid

      call def_var_pnetcdf(ncid,'alon',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'alat',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)

      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)

     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
      call def_var_pnetcdf(ncid,'totx',3,vdims,totx_varid,
     $                     'Net DepthAve x-mom','m/s^2',
     $                     'east_u north_u',.true.)
      call def_var_pnetcdf(ncid,'toty',3,vdims,toty_varid,
     $                     'Net DepthAve y-mom','m/s^2',
     $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'celg',3,vdims,celg_varid,
     $                     'Curl_gGradEl','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'ctsurf',3,vdims,ctsurf_varid,
     $                     '-Curl_DepthAveWindStrs','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'ctbot',3,vdims,ctbot_varid,
     $                     '+Curl_DepthAveBotmStrs','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cpvf',3,vdims,cpvf_varid,
     $                     'TrnsDotGradf/H','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cjbar',3,vdims,cjbar_varid,
     $                     '-J(Kai,1/H)','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cadv',3,vdims,cadv_varid,
     $                     '+Curl_DepthAveHorAdvDif','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cten',3,vdims,cten_varid,
     $                     'dDepthAveVor/dt','1/s^2',
     $                     'east_c north_c',.true.)

! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
!      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm

      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)

      else

!     open netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
      status=nfmpi_open(pom_comm,trim(netcdf_out_file),nf_write,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_open: '//netcdf_out_file,
     $                          status,nf_noerr)

!!     inquire dimensions
!
!      status=nfmpi_inq_dimid(ncid,'time',time_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: time',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'z',z_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: z',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'y',y_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: y',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'x',x_dimid)
!     call handle_error_pnetcdf('nfmpi_inq_dim: x',status,nf_noerr)



!     inqure variables
      status=nfmpi_inq_varid(ncid,'time',time_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: time'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'totx',totx_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: totx'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'toty',toty_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: toty'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'celg',celg_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: celg'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ctsurf',ctsurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ctsurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ctbot',ctbot_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ctbot'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cpvf',cpvf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cpvf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cjbar',cjbar_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cjbar'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cadv',cadv_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cadv'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cten',cten_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cten'
     $                          ,status,nf_noerr)

      endif

!     write data

      start(1)=ioutv
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=ioutv
      edge(1)=im
      edge(2)=jm
      edge(3)=1
      status=nfmpi_put_vara_real_all(
     $      ncid,totx_varid,start,edge,totx)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: totx',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,toty_varid,start,edge,toty)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: toty',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,celg_varid,start,edge,celg)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: celg',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,ctsurf_varid,start,edge,ctsurf)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: ctsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,ctbot_varid,start,edge,ctbot)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: ctbot',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cpvf_varid,start,edge,cpvf)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cpvf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cjbar_varid,start,edge,cjbar)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cjbar',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cadv_varid,start,edge,cadv)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cadv',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cten_varid,start,edge,cten)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cten',status,nf_noerr)
 

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________
!lyo:!vort:end:Write vorticity(e.g. JEBAR) analysis
!_______________________________________________________________________
!lyo:!vort:beg:Write vorticity(e.g. JEBAR) analysis
!_______________________________________________________________________
      subroutine write_vort_pnetcdf( netcdf_out_file )
      implicit none
# include "mpif.h"
# include "pnetcdf.inc"
      include 'pom.h'
      character(len=*), intent(in) :: netcdf_out_file
      character*120 str_tmp!, netcdf_out_file
      integer time_dimid,x_dimid,y_dimid,z_dimid
      integer z_varid,zz_varid,
     $        east_e_varid,
     $        north_e_varid,
     $        h_varid,fsm_varid
      integer time_varid
!     integer totx_varid,toty_varid,celg_varid,ctsurf_varid
      integer celg_varid,ctsurf_varid
      integer ctbot_varid,cpvf_varid,cjbar_varid,cadv_varid,cten_varid
      integer ncid,status
      integer vdims(4)
      integer(mpi_offset_kind) length
      integer(mpi_offset_kind) start(4),edge(4)
      

      ioutv = ioutv + 1

      if ( ioutv == 1  ) then

!     create netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
         status=nfmpi_create(pom_comm,trim(netcdf_out_file),
     $     nf_clobber+nf_64bit_offset,mpi_info_null,ncid) !lyo:pac10:
         call handle_error_pnetcdf('nf_create: '//netcdf_out_file,
     $                          status,nf_noerr)

! define global attributes
      length=len(trim(title))
      status=nfmpi_put_att_text(ncid,nf_global,'title',length,
     $                          trim(title))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

      str_tmp='output file'
      length=len(trim(str_tmp))
      status=nfmpi_put_att_text(ncid,nf_global,'description',length,
     $                          trim(str_tmp))
      call handle_error_pnetcdf('nf_put_att_text',status,nf_noerr)

! define dimensions
      length=nf_unlimited
      status=nfmpi_def_dim(ncid,'time',length,time_dimid)
      call handle_error_pnetcdf('nf_def_dim: time',status,nf_noerr)
      length=kb
      status=nfmpi_def_dim(ncid,'z',length,z_dimid)
      call handle_error_pnetcdf('nf_def_dim: z',status,nf_noerr)
      length=jm_global
      status=nfmpi_def_dim(ncid,'y',length,y_dimid)
      call handle_error_pnetcdf('nf_def_dim: y',status,nf_noerr)
      length=im_global
      status=nfmpi_def_dim(ncid,'x',length,x_dimid)
      call handle_error_pnetcdf('nf_def_dim: x',status,nf_noerr)

! define variables and their attributes
      vdims(1)=time_dimid
      str_tmp='days since '//time_start
      call def_var_pnetcdf(ncid,'time',1,vdims,time_varid,
     $                     'time',str_tmp,
     $                     ' ',.false.)

      vdims(1)=z_dimid
      call def_var_pnetcdf(ncid,'z',1,vdims,z_varid,
     $                     'sigma of cell face','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,z_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=26
      status=nfmpi_put_att_text(ncid,z_varid,'formula_terms',
     $                          length,'sigma: z eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: z',status,nf_noerr)

      call def_var_pnetcdf(ncid,'zz',1,vdims,zz_varid,
     $                     'sigma of cell centre','sigma_level',
     $                     ' ',.false.)
      length=22
      status=nfmpi_put_att_text(ncid,zz_varid,'standard_name',
     $                          length,'ocean_sigma_coordinate')
      length=27
      status=nfmpi_put_att_text(ncid,zz_varid,'formula_terms',
     $                          length,'sigma: zz eta: elb depth: h')
      call handle_error_pnetcdf('nf_put_att_text: zz',status,nf_noerr)

      vdims(1)=x_dimid
      vdims(2)=y_dimid

      call def_var_pnetcdf(ncid,'alon',2,vdims,east_e_varid,
     $                     'easting of elevation points','degree',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'alat',2,vdims,north_e_varid,
     $                     'northing of elevation points','degree',
     $                     'east_e north_e',.true.)

      call def_var_pnetcdf(ncid,'h',2,vdims,h_varid,
     $                     'undisturbed water depth','metre',
     $                     'east_e north_e',.true.)
      call def_var_pnetcdf(ncid,'fsm',2,vdims,fsm_varid,
     $                     'free surface mask','dimensionless',
     $                     'east_e north_e',.true.)

     
      vdims(1)=x_dimid
      vdims(2)=y_dimid
      vdims(3)=time_dimid
!     call def_var_pnetcdf(ncid,'totx',3,vdims,totx_varid,
!    $                     'Net DepthAve x-mom','m/s^2',
!    $                     'east_u north_u',.true.)
!     call def_var_pnetcdf(ncid,'toty',3,vdims,toty_varid,
!    $                     'Net DepthAve y-mom','m/s^2',
!    $                     'east_v north_v',.true.)
      call def_var_pnetcdf(ncid,'celg',3,vdims,celg_varid,
     $                     'Curl_gGradEl','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'ctsurf',3,vdims,ctsurf_varid,
     $                     '-Curl_DepthAveWindStrs','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'ctbot',3,vdims,ctbot_varid,
     $                     '+Curl_DepthAveBotmStrs','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cpvf',3,vdims,cpvf_varid,
     $                     'TrnsDotGradf/H','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cjbar',3,vdims,cjbar_varid,
     $                     '-J(Kai,1/H)','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cadv',3,vdims,cadv_varid,
     $                     '+Curl_DepthAveHorAdvDif','1/s^2',
     $                     'east_c north_c',.true.)
      call def_var_pnetcdf(ncid,'cten',3,vdims,cten_varid,
     $                     'dDepthAveVor/dt','1/s^2',
     $                     'east_c north_c',.true.)

! end definitions
      status=nfmpi_enddef(ncid)
      call handle_error_pnetcdf('nf_enddef: output',status,nf_noerr)

! write data
!      start(1)=1
!      edge(1)=1
!      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
!      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=1
      edge(1)=kb
      status=nfmpi_put_vara_real_all(ncid,z_varid,start,edge,z)
      call handle_error_pnetcdf('nf_put_var_real: z',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,zz_varid,start,edge,zz)
      call handle_error_pnetcdf('nf_put_var_real: zz',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      edge(1)=im
      edge(2)=jm

      status=nfmpi_put_vara_real_all(ncid,east_e_varid,start,edge,
     $                               east_e)
      call handle_error_pnetcdf('nf_put_var_real: east_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,north_e_varid,start,edge,
     $                               north_e)
      call handle_error_pnetcdf('nf_put_var_real: north_e',status,
     $                          nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,h_varid,start,edge,h)
      call handle_error_pnetcdf('nf_put_var_real: h',status,nf_noerr)
      status=nfmpi_put_vara_real_all(ncid,fsm_varid,start,edge,fsm)
      call handle_error_pnetcdf('nf_put_var_real: fsm',status,nf_noerr)

      else

!     open netcdf file
         if(my_task.eq.0) write(*,'(/''writing file '',a)')
     $        trim(netcdf_out_file)
      status=nfmpi_open(pom_comm,trim(netcdf_out_file),nf_write,
     $     mpi_info_null,ncid)
      call handle_error_pnetcdf('nf_open: '//netcdf_out_file,
     $                          status,nf_noerr)

!!     inquire dimensions
!
!      status=nfmpi_inq_dimid(ncid,'time',time_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: time',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'z',z_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: z',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'y',y_dimid)
!      call handle_error_pnetcdf('nfmpi_inq_dim: y',status,nf_noerr)
!      status=nfmpi_inq_dim(ncid,'x',x_dimid)
!     call handle_error_pnetcdf('nfmpi_inq_dim: x',status,nf_noerr)



!     inqure variables
      status=nfmpi_inq_varid(ncid,'time',time_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: time'
     $                          ,status,nf_noerr)
!     status=nfmpi_inq_varid(ncid,'totx',totx_varid)
!     call handle_error_pnetcdf('nfmpi_inq_varid: totx'
!    $                          ,status,nf_noerr)
!     status=nfmpi_inq_varid(ncid,'toty',toty_varid)
!     call handle_error_pnetcdf('nfmpi_inq_varid: toty'
!    $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'celg',celg_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: celg'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ctsurf',ctsurf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ctsurf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'ctbot',ctbot_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: ctbot'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cpvf',cpvf_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cpvf'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cjbar',cjbar_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cjbar'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cadv',cadv_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cadv'
     $                          ,status,nf_noerr)
      status=nfmpi_inq_varid(ncid,'cten',cten_varid)
      call handle_error_pnetcdf('nfmpi_inq_varid: cten'
     $                          ,status,nf_noerr)

      endif

!     write data

      start(1)=ioutv
      edge(1)=1
      status=nfmpi_put_vara_real_all(ncid,time_varid,start,edge,time)
      call handle_error_pnetcdf('nf_put_vara_real:time',status,nf_noerr)

      start(1)=i_global(1)
      start(2)=j_global(1)
      start(3)=ioutv
      edge(1)=im
      edge(2)=jm
      edge(3)=1
!     status=nfmpi_put_vara_real_all(
!    $      ncid,totx_varid,start,edge,totx)
!     call handle_error_pnetcdf(
!    $     'nf_put_vara_real: totx',status,nf_noerr)
!     status=nfmpi_put_vara_real_all(
!    $     ncid,toty_varid,start,edge,toty)
!     call handle_error_pnetcdf(
!    $     'nf_put_vara_real: toty',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,celg_varid,start,edge,celg_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: celg',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $     ncid,ctsurf_varid,start,edge,ctsurf_mean)
      call handle_error_pnetcdf(
     $     'nf_put_vara_real: ctsurf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,ctbot_varid,start,edge,ctbot_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: ctbot',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cpvf_varid,start,edge,cpvf_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cpvf',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cjbar_varid,start,edge,cjbar_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cjbar',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cadv_varid,start,edge,cadv_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cadv',status,nf_noerr)
      status=nfmpi_put_vara_real_all(
     $      ncid,cten_varid,start,edge,cten_mean)
      call handle_error_pnetcdf(
     $      'nf_put_vara_real: cten',status,nf_noerr)
 

! close file
      status=nfmpi_close(ncid)
      call handle_error_pnetcdf('nf_close: output',status,nf_noerr)

      return
      end

!_______________________________________________________________________
!lyo:!vort:end:Write vorticity(e.g. JEBAR) analysis
