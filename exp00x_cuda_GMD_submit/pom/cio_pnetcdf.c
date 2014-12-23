#include<string.h>
#include<stdio.h>
#include<unistd.h>
#include<mpi.h>
#include<pnetcdf.h>

#include"cio_pnetcdf.h"

#define NC_NAME_LEN 120
#define checkPnetcdfErrors(val) \
		handle_error(val, __FILE__, __func__, __LINE__)


static inline char* my_trim(char *src, unsigned len){
	char *ptr = src;
	char *ret = src;
	while (*(ptr++) == (' ')){
		if (ptr-src >= len)
			return NULL;
	}
	ret = ptr-1;
	while (*(ptr++) != (' ')){
		if((ptr-src) >= len)
			return ret;
	}
	*(ptr-1)='\0';
	return ret;
}

static inline void handle_error(int status, const char* file_name,
								const char* func_name, int line_num){
	if (status != NC_NOERR){
		fprintf(stderr, "FILE:%s FUNC:%s LINE:%d\nERRORS:%s\n",
						file_name, func_name, line_num, ncmpi_strerror(status));
		exit(1);
	}
}

/*
void read_grid_pnetcdf_(float *f_z, float *f_zz,
						float f_dx[][i_size], float f_dy[][i_size],
						float f_east_u[][i_size], float f_east_v[][i_size],
						float f_east_e[][i_size], float f_east_c[][i_size],
						float f_north_u[][i_size], float f_north_v[][i_size],
						float f_north_e[][i_size], float f_north_c[][i_size],
						float f_rot[][i_size], float f_h[][i_size],
						float f_fsm[][i_size], float f_dum[][i_size],
						float f_dvm[][i_size]){
*/
void read_grid_pnetcdf(){

	int i, j, k;
	char netcdf_grid_file[120];		
	int z_varid, zz_varid, 
		dx_varid, dy_varid, 
		east_c_varid, east_e_varid, 
		east_u_varid, east_v_varid, 
		north_c_varid, north_e_varid, 
		north_u_varid, north_v_varid, 
		rot_varid, h_varid, fsm_varid,
		dum_varid, dvm_varid;
	
	int ncid; 
	MPI_Offset start[2], count[2];

	strcpy(netcdf_grid_file, "in/");
	strcat(netcdf_grid_file, my_trim(netcdf_file, NC_NAME_LEN));
	strcat(netcdf_grid_file, ".grid.nc");

	if (my_task == master_task){
		printf("\n\nNetcdf file is %s\n\n", netcdf_grid_file);	
	}


	int ret;

//! open netcdf file
	checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_grid_file, NC_NOWRITE,
								  MPI_INFO_NULL, &ncid));

//! get variables
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "z", &z_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "zz", &zz_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "dx", &dx_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "dy", &dy_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "east_u", &east_u_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "east_v", &east_v_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "east_e", &east_e_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "east_c", &east_c_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "north_u", &north_u_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "north_v", &north_v_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "north_e", &north_e_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "north_c", &north_c_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "rot", &rot_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "h", &h_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "fsm", &fsm_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "dum", &dum_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "dvm", &dvm_varid));

	start[0] = 0;
	count[0] = kb;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, z_varid, start, count, z));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, zz_varid, start, count, zz)); 

	start[0] = j_global[0]-1;
	start[1] = i_global[0]-1;
	count[0] = jm;
	count[1] = im;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, dx_varid, start, count, (float*)dx));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, dy_varid, start, count, (float*)dy));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, east_u_varid, start, count, (float*)east_u));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, east_v_varid, start, count, (float*)east_v));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, east_e_varid, start, count, (float*)east_e));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, east_c_varid, start, count, (float*)east_c));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, north_u_varid, start, count, (float*)north_u));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, north_v_varid, start, count, (float*)north_v));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, north_e_varid, start, count, (float*)north_e));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, north_c_varid, start, count, (float*)north_c));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, rot_varid, start, count, (float*)rot));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, h_varid, start, count, (float*)h));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, fsm_varid, start, count, (float*)fsm));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, dum_varid, start, count, (float*)dum));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, dvm_varid, start, count, (float*)dvm));

//! close file:
	checkPnetcdfErrors(ncmpi_close(ncid));

/*
	for (k = 0; k < kb; k++){
		if (my_task == master_task)
			printf("z[%d] = %f\n",k ,z[k]);
		f_z[k] = z[k];
		f_zz[k] = zz[k];
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_dx[j][i] = dx[j][i];	
			f_dy[j][i] = dy[j][i];	
			f_east_u[j][i] = east_u[j][i];	
			f_east_v[j][i] = east_v[j][i];	
			f_east_e[j][i] = east_e[j][i];	
			f_east_c[j][i] = east_c[j][i];	
			f_north_u[j][i] = north_u[j][i];	
			f_north_v[j][i] = north_v[j][i];	
			f_north_e[j][i] = north_e[j][i];	
			f_north_c[j][i] = north_c[j][i];	

			f_rot[j][i] = rot[j][i];	
			f_h[j][i] = h[j][i];	
			f_fsm[j][i] = fsm[j][i];	
			f_dum[j][i] = dum[j][i];	
			f_dvm[j][i] = dvm[j][i];	
		}
	}
*/
	
}


void read_initial_ts_pnetcdf(int k,
							 float temp[][j_size][i_size],
							 float salt[][j_size][i_size]){
	//int k = *f_k;
	char netcdf_ic_file[NC_NAME_LEN];
	int ncid;
	int tb_varid, sb_varid;
	MPI_Offset start[3], count[3];

	strcpy(netcdf_ic_file, "in/");
	strcat(netcdf_ic_file, my_trim(netcdf_file, NC_NAME_LEN));
	strcat(netcdf_ic_file, ".ts_initial.nc");

	if (my_task == master_task){
		printf("\n\nNetcdf ic file is %s\n\n", netcdf_ic_file);	
	}
	
	checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_ic_file, NC_NOWRITE,
								  MPI_INFO_NULL, &ncid));

	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "tb", &tb_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "sb", &sb_varid));


	start[0] = 0;
	start[1] = j_global[0]-1;
	start[2] = i_global[0]-1;

	count[0] = k;
	count[1] = jm;
	count[2] = im;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, tb_varid, start, count, (float*)temp));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, sb_varid, start, count, (float*)salt));


	checkPnetcdfErrors(ncmpi_close(ncid));
	return;
}

void read_mean_ts_pnetcdf(float temp[][j_size][i_size],
						  float salt[][j_size][i_size]){
	//int k = *f_k;
	char netcdf_mean_ts_file[NC_NAME_LEN];
	int ncid;
	int tmean_varid, smean_varid;
	MPI_Offset start[3], count[3];

	strcpy(netcdf_mean_ts_file, "in/");
	strcat(netcdf_mean_ts_file, my_trim(netcdf_file, NC_NAME_LEN));
	strcat(netcdf_mean_ts_file, ".ts_mean.nc");

	if (my_task == master_task){
		printf("\n\nNetcdf ic file is %s\n\n", netcdf_mean_ts_file);	
	}
	
	checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_mean_ts_file, NC_NOWRITE,
								  MPI_INFO_NULL, &ncid));

	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "tmean", &tmean_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "smean", &smean_varid));


	start[0] = 0;
	start[1] = j_global[0]-1;
	start[2] = i_global[0]-1;

	count[0] = kb;
	count[1] = jm;
	count[2] = im;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, tmean_varid, start, count, (float*)temp));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, smean_varid, start, count, (float*)salt));


	checkPnetcdfErrors(ncmpi_close(ncid));
	return;
}

void read_clim_ts_pnetcdf(float temp[][j_size][i_size],
						  float salt[][j_size][i_size]){
	//int k = *f_k;
	char netcdf_clim_ts_file[NC_NAME_LEN];
	int ncid;
	int tclim_varid, sclim_varid;
	MPI_Offset start[3], count[3];

	strcpy(netcdf_clim_ts_file, "in/");
	strcat(netcdf_clim_ts_file, my_trim(netcdf_file, NC_NAME_LEN));
	strcat(netcdf_clim_ts_file, ".ts_mean.nc");

	if (my_task == master_task){
		printf("\n\nNetcdf ic file is %s\n\n", netcdf_clim_ts_file);	
	}
	
	checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_clim_ts_file, NC_NOWRITE,
								  MPI_INFO_NULL, &ncid));

	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "tclim", &tclim_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "sclim", &sclim_varid));


	start[0] = 0;
	start[1] = j_global[0]-1;
	start[2] = i_global[0]-1;

	count[0] = kb;
	count[1] = jm;
	count[2] = im;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, tclim_varid, start, count, (float*)temp));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, sclim_varid, start, count, (float*)salt));


	checkPnetcdfErrors(ncmpi_close(ncid));
	return;
}

void read_tide_east_pnetcdf(float ampe_out[][j_size],
							float phae_out[][j_size],
							float amue_out[][j_size],
							float phue_out[][j_size]){
//! fhx:tide:read tide at the eastern boundary for PROFS

	//int k = *f_k;
	char netcdf_bc_file[NC_NAME_LEN];
	int ncid;
	int ampe_varid, phae_varid;
	int amue_varid, phue_varid;

	MPI_Offset start[2], count[2];

	strcpy(netcdf_bc_file, "in/");
	strcat(netcdf_bc_file, "tide.nc");

	if (my_task == master_task){
		printf("\n\nNetcdf ic file is %s\n\n", netcdf_bc_file);	
	}
	
	checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_bc_file, NC_NOWRITE,
								  MPI_INFO_NULL, &ncid));

	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "ampe", &ampe_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "phae", &phae_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "amue", &amue_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "phue", &phue_varid));


	start[0] = 0;
	start[1] = j_global[0]-1;

	count[0] = ntide;
	count[1] = jm;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, ampe_varid, start, count, (float*)ampe_out));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, phae_varid, start, count, (float*)phae_out));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, amue_varid, start, count, (float*)amue_out));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, phue_varid, start, count, (float*)phue_out));


	checkPnetcdfErrors(ncmpi_close(ncid));
	return;
}


/*
void read_restart_pnetcdf(){
	char netcdf_in_file[NC_NAME_LEN];
	int ncid;
	int ampe_varid, phae_varid;
	int amue_varid, phue_varid;

	MPI_Offset start[2], count[2];

	strcpy(netcdf_in_file, "in/");
	strcat(netcdf_in_file, my_trim(read_rst_file, NC_NAME_LEN));

	if (my_task == master_task){
		printf("\n\nNetcdf ic file is %s\n\n", netcdf_bc_file);	
	}
	
	checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_bc_file, NC_NOWRITE,
								  MPI_INFO_NULL, &ncid));

	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "ampe", &ampe_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "phae", &phae_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "amue", &amue_varid));
	checkPnetcdfErrors(ncmpi_inq_varid(ncid, "phue", &phue_varid));


	start[0] = 0;
	start[1] = j_global[0]-1;

	count[0] = n_tide;
	count[1] = jm;

	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, ampe_varid, start, count, (float*)ampe_out));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, phae_varid, start, count, (float*)phae_out));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, amue_varid, start, count, (float*)amue_out));
	checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, phue_varid, start, count, (float*)phue_out));


	checkPnetcdfErrors(ncmpi_close(ncid));
	return;
}
*/


void read_uabe_pnetcdf(){

//!lyo:pac10:use stcc version
//! read transport at the eastern boundary for PROFS in GOM !ayumi 2010/4/7
//!lyo:20110224:alu:stcc:modified to work if *bc.nc* is not provided
//!     as in STCC idealized run

	char netcdf_in_file[NC_NAME_LEN];
	int ncid;
	int uabe_varid;
	int j;

	MPI_Offset start[2], count[2];

	strcpy(netcdf_in_file, "in/");
	strcat(netcdf_in_file, "bc.nc");

	if (my_task == master_task){
		printf("\n\nNetcdf ic file is %s\n\n", netcdf_in_file);	
	}
	
	if (access(netcdf_in_file, F_OK) == 0){
		checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_in_file, NC_NOWRITE,
									  MPI_INFO_NULL, &ncid));

		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "uabe", &uabe_varid));

		start[0] = j_global[0]-1;
		count[0] = jm;

		checkPnetcdfErrors(ncmpi_get_vara_float_all(ncid, uabe_varid, start, count, (float*)uabe));


		checkPnetcdfErrors(ncmpi_close(ncid));
	}else{
         printf("****Warning from subr. read_uabe_pnetcdf****\n");
         printf("uabe file = %s\n", netcdf_in_file);
         printf("does not exist, and uabe set to zero\n");

		 for (j = 0; j < jm; j++){
			uabe[j] = 0.0;
		 }
	}
	return;
}


void def_var_pnetcdf(const int ncid, const char *name, 
					 const int nvdims, const int *vdims,
					 int *varid, 
					 const char *long_name,
					 const char *units,
					 const char *coords,
					 int lcoords){

	MPI_Offset length;
	int varid_now;

	checkPnetcdfErrors(ncmpi_def_var(ncid, name, NC_FLOAT, 
									 nvdims, vdims, varid));
	length = strlen(long_name);
	varid_now = *varid;	
	checkPnetcdfErrors(ncmpi_put_att_text(ncid, varid_now, 
										  "long_name", length, long_name));
	length = strlen(units);
	checkPnetcdfErrors(ncmpi_put_att_text(ncid, varid_now,
										  "units", length, units));
	if (lcoords){
		length = strlen(coords);	
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, varid_now,
											"coordinates", length, coords));
	}

}

void *write_output_ptr(int num_out){
	  write_output(num_out);
}

void write_output(int num_out){
	
	int i, j, k;
	float rdisp;

	if (strcmp(netcdf_file, "nonetcdf") != 0){
		//rdisp = 1.f/num;	
		rdisp = 1.f/iprint;	

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				uab_mean[j][i] *= rdisp;	
				vab_mean[j][i] *= rdisp;	
				elb_mean[j][i] *= rdisp;	

				wusurf_mean[j][i] *= rdisp;	
				wvsurf_mean[j][i] *= rdisp;	
				wtsurf_mean[j][i] *= rdisp;	
				wssurf_mean[j][i] *= rdisp;	

			}
		}

		for (k = 0; k < kb; k++){
			for (j = 0; j < jm; j++){
				for (i = 0; i < im; i++){
					 u_mean[k][j][i] *= rdisp;
					 v_mean[k][j][i] *= rdisp;
					 w_mean[k][j][i] *= rdisp;
					 t_mean[k][j][i] *= rdisp;
					 s_mean[k][j][i] *= rdisp;
					 rho_mean[k][j][i] *= rdisp;
					 kh_mean[k][j][i] *= rdisp;
					 km_mean[k][j][i] *= rdisp;
				}
			}
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				w_mean[kb-1][j][i] = ustks[0][j][i];	
				kh_mean[kb-1][j][i] = vstks[0][j][i];	
			}
		}

		if (calc_assim){
			printf("It is not supported now! FILE:%s, LINE:%d\n",
				   __FILE__, __LINE__);	
		}

		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				s_mean[kb-1][j][i] = ssurf[j][i];	
				rho_mean[kb-1][j][i] = tsurf[j][i];	
			}
		}


		exchange2d_mpi_tag(uab_mean, im, jm, 10);
		exchange2d_mpi_tag(vab_mean, im, jm, 10);
		exchange2d_mpi_tag(elb_mean, im, jm, 10);
		exchange2d_mpi_tag(wusurf_mean, im, jm, 10);
		exchange2d_mpi_tag(wvsurf_mean, im, jm, 10);
		exchange2d_mpi_tag(wtsurf_mean, im, jm, 10);
		exchange2d_mpi_tag(wssurf_mean, im, jm, 10);

		exchange2d_mpi_tag(et, im, jm, 10);

		exchange3d_mpi_tag(u_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(v_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(w_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(t_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(s_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(rho_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(kh_mean, im, jm, kb, 10);
		exchange3d_mpi_tag(km_mean, im, jm, kb, 10);


		if (output_flag == 1){
			write_output_pnetcdf(num_out);	
		}else if (output_flag == 0){
			printf("It is not supported now! FILE:%s, LINE:%d\n",
				   __FILE__, __LINE__);	
			//write_output_pnetcdf0();	
		}

		//for (j = 0; j < jm; j++){
		//	for (i = 0; i < im; i++){
		//		uab_mean[j][i] = 0;	
		//		vab_mean[j][i] = 0;	
		//		elb_mean[j][i] = 0;	

		//		wusurf_mean[j][i] = 0;	
		//		wvsurf_mean[j][i] = 0;	
		//		wtsurf_mean[j][i] = 0;	
		//		wssurf_mean[j][i] = 0;	

		//	}
		//}

		//for (k = 0; k < kb; k++){
		//	for (j = 0; j < jm; j++){
		//		for (i = 0; i < im; i++){
		//			 u_mean[k][j][i] = 0;
		//			 v_mean[k][j][i] = 0;
		//			 w_mean[k][j][i] = 0;
		//			 t_mean[k][j][i] = 0;
		//			 s_mean[k][j][i] = 0;
		//			 rho_mean[k][j][i] = 0;
		//			 kh_mean[k][j][i] = 0;
		//			 km_mean[k][j][i] = 0;
		//		}
		//	}
		//}
		num = 0;
	}

	return;
}

/*
      subroutine write_output_surf( d_in )
      use module_time

      implicit none
      include 'pom.h'

      type(date), intent(in) :: d_in

      integer i,j,k
      real u_tmp, v_tmp, rdisp

      if(netcdf_file.ne.'nonetcdf' .and. mod(iint,iprints).eq.0) then

         rdisp = 1./ real ( nums ) !lyo:vort:
         usrf_mean    = usrf_mean *rdisp
         vsrf_mean    = vsrf_mean *rdisp
         elsrf_mean    = elsrf_mean *rdisp
         uwsrf_mean = uwsrf_mean *rdisp
         vwsrf_mean = vwsrf_mean *rdisp

         utf_mean    = utf_mean *rdisp !lyo:!vort:U-transport!part of CPVF:
         vtf_mean    = vtf_mean *rdisp !Should be (ua,va)D@isplit; but approx.ok

!     fill up ghost cells before output
         call exchange2d_mpi( usrf_mean, im, jm )
         call exchange2d_mpi( vsrf_mean, im, jm )
         call exchange2d_mpi( elsrf_mean, im, jm )
         call exchange2d_mpi( uwsrf_mean, im, jm )
         call exchange2d_mpi( vwsrf_mean, im, jm )
       
         call exchange2d_mpi( utf_mean, im, jm )
         call exchange2d_mpi( vtf_mean, im, jm )

       if (SURF_flag==1) then
!
          if ( calc_stokes ) then
             xstks_mean = xstks_mean *rdisp
             ystks_mean = ystks_mean *rdisp
             call exchange3d_mpi( xstks_mean, im, jm, kb )
             call exchange3d_mpi( ystks_mean, im, jm, kb )
!            call write_SURFStokes_pnetcdf( !exp347:turn off
             call write_SURF_pnetcdf( 
     $        "out/SRF."//trim(netcdf_file)//".nc")
             else
!
             call write_SURF_pnetcdf( 
     $        "out/SRF."//trim(netcdf_file)//".nc")
             endif
!
!lyo:!vort:beg:Write vorticity(e.g. JEBAR) analysis
          if ( calc_vort ) then
      celg_mean = celg_mean*rdisp; ctsurf_mean=ctsurf_mean*rdisp;
      ctbot_mean=ctbot_mean*rdisp; cpvf_mean  =  cpvf_mean*rdisp;
      cjbar_mean=cjbar_mean*rdisp; cadv_mean  =  cadv_mean*rdisp;
      cten_mean = cten_mean*rdisp;
             call write_vort_pnetcdf( 
     $        "out/vor."//trim(netcdf_file)//".nc")
      CELG_mean=0.0; CTSURF_mean=0.0; CTBOT_mean=0.0; CPVF_mean=0.0; 
      CJBAR_mean=0.0; CADV_mean=0.0; CTEN_mean=0.0; 
             endif
!lyo:!vort:end:Write vorticity(e.g. JEBAR) analysis
!
          endif
       
         usrf_mean    = 0.0
         vsrf_mean    = 0.0
         elsrf_mean    = 0.0
         uwsrf_mean = 0.0
         vwsrf_mean = 0.0

         utf_mean    = 0.0
         vtf_mean    = 0.0

         xstks_mean = 0.0
         ystks_mean = 0.0

         nums = 0

      endif

      return
      end
*/


/*
void write_output_surf(){
	
	int i, j, k;
	float rdisp;
	if (strcmp(netcdf_file, "nonetcdf") != 0 &&
		(iint%iprints) != 0){
		
		rdisp = 1.f/nums;
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				usrf_mean[j][i] *= rdisp;	
				vsrf_mean[j][i] *= rdisp;	
				elsrf_mean[j][i] *= rdisp;	
				uwsrf_mean[j][i] *= rdisp;	
				vwsrf_mean[j][i] *= rdisp;	

				utf_mean[j][i] *= rdisp;	
				vtf_mean[j][i] *= rdisp;	
			}
		}

		exchange2d_mpi(usrf_mean, im, jm);
		exchange2d_mpi(vsrf_mean, im, jm);
		exchange2d_mpi(elsrf_mean, im, jm);
		exchange2d_mpi(uwsrf_mean, im, jm);
		exchange2d_mpi(vwsrf_mean, im, jm);

		exchange2d_mpi(utf_mean, im, jm);
		exchange2d_mpi(vtf_mean, im, jm);

		if (SURF_flag == 1){
			if (calc_stokes){
				for (k = 0; k < kb; k++){
					for (j = 0; j < jm; j++){
						for (i = 0; i < im; i++){
							xstks[k][j][i] *= rdisp;	
							ystks[k][j][i] *= rdisp;	
						}
					}
				}

				exchange3d_mpi(xstks_mean, im, jm, kb);
				exchange3d_mpi(ystks_mean, im, jm, kb);

				write_SURF_pnetcdf();
			}else{
				write_SURF_pnetcdf();
			}

			if (calc_vort){
				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						celg_mean[j][i] *= rdisp;
						ctsurf_mean[j][i] *= rdisp;
						ctbot_mean[j][i] *= rdisp;
						cpvf_mean[j][i] *= rdisp;
						cjbar_mean[j][i] *= rdisp;
						cadv_mean[j][i] *= rdisp;
						cten_mean[j][i] *= rdisp;
					}
				}

				write_vort_pnetcdf();

				for (j = 0; j < jm; j++){
					for (i = 0; i < im; i++){
						celg_mean[j][i] = 0;
						ctsurf_mean[j][i] = 0;
						ctbot_mean[j][i] = 0;
						cpvf_mean[j][i] = 0;
						cjbar_mean[j][i] = 0;
						cadv_mean[j][i] = 0;
						cten_mean[j][i] = 0;
					}
				}
			}
		}
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			usrf_mean[j][i] = 0;	
			vsrf_mean[j][i] = 0;	
			elsrf_mean[j][i] = 0;	
			uwsrf_mean[j][i] = 0;	
			vwsrf_mean[j][i] = 0;	

			utf_mean[j][i] = 0;	
			vtf_mean[j][i] = 0;	

			xstks_mean[j][i] = 0;	
			ystks_mean[j][i] = 0;	
		}
	}

	nums = 0;
}
*/


void write_output_pnetcdf_bak(){

	int i, j, k;
	char netcdf_output_file[120];		
	char time_start_output[120];		

	int z_varid, zz_varid, 
		dx_varid, dy_varid, 
		east_u_varid, east_v_varid, 
		east_c_varid, east_e_varid, 
		north_u_varid, north_v_varid, 
		north_c_varid, north_e_varid, 
		rot_varid, h_varid, fsm_varid,
		dum_varid, dvm_varid;

	int uab_varid, vab_varid, elb_varid;
    int wusurf_varid,wvsurf_varid,wtsurf_varid,wssurf_varid;
	int u_varid,v_varid, w_varid,t_varid,s_varid,
		rho_varid,km_varid,kh_varid;
    int tr_varid;  //!fhx:tracer
	int time_varid;
    //int time_varid,u_varid,v_varid,
    //     w_varid,t_varid,s_varid,rho_varid,km_varid,kh_varid;
    //int tr_varid;  //!fhx:tracer

	int time_dimid, z_dimid, y_dimid, x_dimid, nb_dimid;
	int vdims[5];
	
	int ncid; 
	MPI_Offset start[5], count[5];
	MPI_Offset length;

	strcpy(netcdf_output_file, "out/");
	strcat(netcdf_output_file, my_trim(netcdf_file, NC_NAME_LEN));
	strcat(netcdf_output_file, ".nc");

	if (my_task == master_task){
		printf("\n\nNetcdf file is %s\n\n", netcdf_output_file);	
	}

	num_out += 1;

	if (num_out == 1){
		if (my_task == master_task){
			printf("\n\n Creating Netcdf file: %s\n\n", netcdf_output_file);	
		}
		//it is the first IO, we need to create related netcdf files	
		checkPnetcdfErrors(ncmpi_create(pom_comm, netcdf_output_file, 
									    NC_64BIT_OFFSET,
									    MPI_INFO_NULL, &ncid));

		length = strlen(my_trim(title, 40));
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, NC_GLOBAL, "title",
											  length, my_trim(title, 40)));

		char *str_tmp = "output_file";
		length = strlen(str_tmp);
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, NC_GLOBAL, "description",
											  length, str_tmp));


		checkPnetcdfErrors(ncmpi_def_dim(ncid, "time", NC_UNLIMITED, 
										 &time_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "z", kb, 
										 &z_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "y", jm_global, 
										 &y_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "x", im_global, 
										 &x_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "nb", nb, 
										 &nb_dimid));

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//1-D variable definations
		/*
		void def_var_pnetcdf(const int ncid, const char *name, 
							 const int nvdims, const int *vdims,
							 int *varid, const char *long_name,
							 const char *units){
		*/
		vdims[0] = time_dimid;
		strcpy(time_start_output, "days_since");
		strcat(time_start_output, my_trim(time_start, 26));
		def_var_pnetcdf(ncid, "time", 1, vdims, &time_varid,
						"time", time_start_output, " ", 0);

		vdims[0] = z_dimid;

		//////////////////////////////////////////////////////////////////
		def_var_pnetcdf(ncid, "z", 1, vdims, &z_varid,
						"sigma of cell face", "sigma_level", " ", 0);
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, z_varid, 
											  "standard_name",
											  22, "ocean_sigma_coordinate"));		
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, z_varid, 
											  "formula_terms",
											  26, "sigma: z eta:elb depth:h"));		
		//////////////////////////////////////////////////////////////////
		def_var_pnetcdf(ncid, "zz", 1, vdims, &zz_varid, 
						"sigma of cell centre", "sigma_level", " ", 0);

		checkPnetcdfErrors(ncmpi_put_att_text(ncid, zz_varid, 
											  "standard_name",
											  22, "ocean_sigma_coordinate"));

		checkPnetcdfErrors(ncmpi_put_att_text(ncid, zz_varid, 
											  "formula_terms",
											  27, "sigma:z eta:elb depth:h"));

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//2-D variable definations
		vdims[1] = x_dimid;
		vdims[0] = y_dimid;
		def_var_pnetcdf(ncid, "dx", 2, vdims, &dx_varid,
						"grid increment in x", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "dy", 2, vdims, &dy_varid,
						"grid increment in y", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "east_u", 2, vdims, &east_u_varid,
						"easting of u-points", "degree",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "east_v", 2, vdims, &east_v_varid,
						"easting of v-points", "degree",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "east_e", 2, vdims, &east_e_varid,
						"easting of elevation points", "degree",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "east_c", 2, vdims, &east_c_varid,
						"easting of cell cormers", "degree",
						"east_c north_c", 1);
		def_var_pnetcdf(ncid, "north_u", 2, vdims, &north_u_varid,
						"northing of u-points", "degree",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "north_v", 2, vdims, &north_v_varid,
						"northing of v-points", "degree",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "north_e", 2, vdims, &north_e_varid,
						"northing of elevation points", "degree",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "north_c", 2, vdims, &north_c_varid,
						"northing of cell corners", "degree",
						"east_c north_c", 1);
		def_var_pnetcdf(ncid, "rot", 2, vdims, &rot_varid,
						"Rotation angle of x-axis wrt. east", "degree",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "h", 2, vdims, &h_varid,
						"undisturbed water depth", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "fsm", 2, vdims, &fsm_varid,
						"free surface mask", "dimensionless",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "dum", 2, vdims, &dum_varid,
						"u-velocity mask", "dimensionless",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "dvm", 2, vdims, &dvm_varid,
						"v-velocity mask", "dimensionless",
						"east_v north_v", 1);

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//3-D variable definations
		vdims[2] = x_dimid;
		vdims[1] = y_dimid;
		vdims[0] = time_dimid;

		def_var_pnetcdf(ncid, "uab", 3, vdims, &uab_varid,
						"depth-averaged u", "metre/sec",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "vab", 3, vdims, &vab_varid,
						"depth-averaged v", "metre/sec",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "elb", 3, vdims, &elb_varid,
						"surface elevation", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "wusurf", 3, vdims, &wusurf_varid,
						"x-momentum flux", "metre^2/sec^2",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "wvsurf", 3, vdims, &wvsurf_varid,
						"y-momentum flux", "metre^2/sec^2",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "wtsurf", 3, vdims, &wtsurf_varid,
						"temperature flux", "deg m/s",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "wssurf", 3, vdims, &wssurf_varid,
						"salinity flux", "psu m/s",
						"east_e north_e", 1);

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//4-D variable definations
		vdims[3] = x_dimid;
		vdims[2] = y_dimid;
		vdims[1] = z_dimid;
		vdims[0] = time_dimid;

		def_var_pnetcdf(ncid, "u", 4, vdims, &u_varid,
						"x-velocity", "metre/sec",
						"east_u north_u zz", 1);
		def_var_pnetcdf(ncid, "v", 4, vdims, &v_varid,
						"y-velocity", "metre/sec",
						"east_v north_v zz", 1);
		def_var_pnetcdf(ncid, "w", 4, vdims, &w_varid,
						"z-velocity", "metre/sec",
						"east_e north_e z", 1);
		def_var_pnetcdf(ncid, "t", 4, vdims, &t_varid,
						"potential temperature", "K",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "s", 4, vdims, &s_varid,
						"salinity x rho / rhoref", "PSS",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "rho", 4, vdims, &rho_varid,
						"(density-1000)/rhoref", "dimensionless",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "kh", 4, vdims, &kh_varid,
						"vertical diffusivity", "metre2/sec",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "km", 4, vdims, &km_varid,
						"vertical viscosity", "metre2/sec",
						"east_e north_e zz", 1);
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//5-D variable definations
		vdims[4] = x_dimid;
		vdims[3] = y_dimid;
		vdims[2] = z_dimid;
		vdims[1] = nb_dimid;
		vdims[0] = time_dimid;

		def_var_pnetcdf(ncid, "tr", 5, vdims, &tr_varid,
						"tracer", "dimensionless",
						"east_u north_u zz nb", 1);

		checkPnetcdfErrors(ncmpi_enddef(ncid));

		///////////////////////////////////////////////////////////
		//1-D output	
		start[0] = 0;
		count[0] = kb;
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, z_varid, 
													start, count, z));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, zz_varid, 
													start, count, zz));
		///////////////////////////////////////////////////////////
		//2-D output	
		start[1] = i_global[0]-1;
		start[0] = j_global[0]-1;
		count[1] = im;
		count[0] = jm;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dx_varid,
											start, count, (float*)dx));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dy_varid,
											start, count, (float*)dy));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_u_varid,
											start, count, (float*)east_u));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_v_varid,
											start, count, (float*)east_v));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_e_varid,
											start, count, (float*)east_e));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_c_varid,
											start, count, (float*)east_c));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_u_varid,
											start, count, (float*)north_u));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_v_varid,
											start, count, (float*)north_v));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_e_varid,
											start, count, (float*)north_e));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_c_varid,
											start, count, (float*)north_c));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, rot_varid,
											start, count, (float*)rot));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, h_varid,
											start, count, (float*)h));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, fsm_varid,
											start, count, (float*)fsm));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dum_varid,
											start, count, (float*)dum));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dvm_varid,
											start, count, (float*)dvm));

		///////////////////////////////////////////////////////////
		//3-D output	
		start[2] = i_global[0]-1;
		start[1] = j_global[0]-1;
		start[0] = 0;

		count[2] = im;
		count[1] = jm;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, uab_varid,
											start, count, (float*)uab));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, vab_varid,
											start, count, (float*)vab));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, elb_varid,
											start, count, (float*)elb));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wusurf_varid,
											start, count, (float*)wusurf));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wvsurf_varid,
											start, count, (float*)wvsurf));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wtsurf_varid,
											start, count, (float*)wtsurf));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wssurf_varid,
											start, count, (float*)wssurf));

		///////////////////////////////////////////////////////////
		//4-D output	
		start[3] = i_global[0]-1;
		start[2] = j_global[0]-1;
		start[1] = 0;
		start[0] = 0;

		count[3] = im;
		count[2] = jm;
		count[1] = kb;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, u_varid,
											start, count, (float*)u));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, v_varid,
											start, count, (float*)v));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, w_varid,
											start, count, (float*)w));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, t_varid,
											start, count, (float*)t));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, s_varid,
											start, count, (float*)s));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, rho_varid,
											start, count, (float*)rho));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, kh_varid,
											start, count, (float*)kh));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, km_varid,
											start, count, (float*)km));

		///////////////////////////////////////////////////////////
		//5-D output	
		start[4] = i_global[0]-1;
		start[3] = j_global[0]-1;
		start[2] = 0;
		start[1] = 0;
		start[0] = 0;

		count[4] = im;
		count[3] = jm;
		count[2] = kb;
		count[1] = nb;
		count[1] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, tr_varid,
											start, count, (float*)tr));

	
	}else{
		if (my_task == master_task){
			printf("\n\n Openning Netcdf file: %s\n\n", netcdf_output_file);	
		}

		checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_output_file,
									  NC_WRITE, MPI_INFO_NULL, &ncid));

		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "time", &time_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "uab",  &uab_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "vab",  &vab_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "elb",  &elb_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wusurf", &wusurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wvsurf", &wvsurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wtsurf", &wtsurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wssurf", &wssurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "u", &u_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "v", &v_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "w", &w_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "t", &t_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "s", &s_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "rho", &rho_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "kh", &kh_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "km", &km_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "tr", &tr_varid));

		///////////////////////////////////////////////////////////
		//1-D output	
		start[0] = num_out-1;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, time_varid, 
										start, count, (float*)&model_time));

		///////////////////////////////////////////////////////////
		//3-D output	
		start[2] = i_global[0]-1;
		start[1] = j_global[0]-1;
		start[0] = num_out-1;

		count[2] = im;
		count[1] = jm;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, uab_varid, 
										start, count, (float*)uab_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, vab_varid, 
										start, count, (float*)vab_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, elb_varid, 
										start, count, (float*)elb_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wusurf_varid, 
										start, count, (float*)wusurf_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wvsurf_varid, 
										start, count, (float*)wvsurf_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wtsurf_varid, 
										start, count, (float*)wtsurf_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wssurf_varid, 
										start, count, (float*)wssurf_mean));

		///////////////////////////////////////////////////////////
		//4-D output	
		start[3] = i_global[0]-1;
		start[2] = j_global[0]-1;
		start[1] = 0;
		start[0] = num_out-1;

		count[3] = im;
		count[2] = jm;
		count[1] = kb;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, u_varid, 
										start, count, (float*)u_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, v_varid, 
										start, count, (float*)v_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, w_varid, 
										start, count, (float*)w_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, t_varid, 
										start, count, (float*)t_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, s_varid, 
										start, count, (float*)s_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, rho_varid, 
										start, count, (float*)rho_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, kh_varid, 
										start, count, (float*)kh_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, km_varid, 
										start, count, (float*)km_mean));

		///////////////////////////////////////////////////////////
		//5-D output	
		start[4] = i_global[0]-1;
		start[3] = j_global[0]-1;
		start[2] = 0;
		start[1] = 0;
		start[0] = num_out-1;

		count[4] = im;
		count[3] = jm;
		count[2] = kb;
		count[1] = nb;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, tr_varid, 
										start, count, (float*)tr));
	}

	checkPnetcdfErrors(ncmpi_close(ncid));
	return;


/*
	for (k = 0; k < kb; k++){
		if (my_task == master_task)
			printf("z[%d] = %f\n",k ,z[k]);
		f_z[k] = z[k];
		f_zz[k] = zz[k];
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_dx[j][i] = dx[j][i];	
			f_dy[j][i] = dy[j][i];	
			f_east_u[j][i] = east_u[j][i];	
			f_east_v[j][i] = east_v[j][i];	
			f_east_e[j][i] = east_e[j][i];	
			f_east_c[j][i] = east_c[j][i];	
			f_north_u[j][i] = north_u[j][i];	
			f_north_v[j][i] = north_v[j][i];	
			f_north_e[j][i] = north_e[j][i];	
			f_north_c[j][i] = north_c[j][i];	

			f_rot[j][i] = rot[j][i];	
			f_h[j][i] = h[j][i];	
			f_fsm[j][i] = fsm[j][i];	
			f_dum[j][i] = dum[j][i];	
			f_dvm[j][i] = dvm[j][i];	
		}
	}
*/
}

void write_output_pnetcdf(int num_out){

	int i, j, k;
	char netcdf_output_file[120];		
	char time_start_output[120];		

	int z_varid, zz_varid, 
		dx_varid, dy_varid, 
		east_u_varid, east_v_varid, 
		east_c_varid, east_e_varid, 
		north_u_varid, north_v_varid, 
		north_c_varid, north_e_varid, 
		rot_varid, h_varid, fsm_varid,
		dum_varid, dvm_varid;

	int uab_varid, vab_varid, et_varid, elb_varid;
    int wusurf_varid,wvsurf_varid,wtsurf_varid,wssurf_varid;
	int u_varid,v_varid, w_varid,t_varid,s_varid,
		rho_varid,km_varid,kh_varid;
    int tr_varid;  //!fhx:tracer
	int time_varid;
    //int time_varid,u_varid,v_varid,
    //     w_varid,t_varid,s_varid,rho_varid,km_varid,kh_varid;
    //int tr_varid;  //!fhx:tracer

	int time_dimid, z_dimid, y_dimid, x_dimid, nb_dimid;
	int vdims[5];
	
	int ncid; 
	MPI_Offset start[5], count[5];
	MPI_Offset length;

	//if (my_task == master_task){
	//	printf("\n\nBefore Netcdf file is %s\n\n", netcdf_output_file);	
	//}

	strcpy(netcdf_output_file, "out/");
	strcat(netcdf_output_file, my_trim(netcdf_file, NC_NAME_LEN));
	strcat(netcdf_output_file, ".nc");

	if (my_task == master_task){
		printf("\n\nNetcdf file is %s\n\n", netcdf_output_file);	
	}

	//num_out += 1;

	if (num_out == 0){
		//if (my_task == master_task){
			printf("\n\n Creating Netcdf file: %saaa, num_out:%d\n\n", 
					netcdf_output_file, num_out);	
		//}
		//it is the first IO, we need to create related netcdf files	
		checkPnetcdfErrors(ncmpi_create(pom_comm, netcdf_output_file, 
									    NC_64BIT_OFFSET,
									    MPI_INFO_NULL, &ncid));

		length = strlen(my_trim(title, 40));
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, NC_GLOBAL, "title",
											  length, my_trim(title, 40)));

		char *str_tmp = "output_file";
		length = strlen(str_tmp);
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, NC_GLOBAL, "description",
											  length, str_tmp));


		checkPnetcdfErrors(ncmpi_def_dim(ncid, "time", NC_UNLIMITED, 
										 &time_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "z", kb, 
										 &z_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "y", jm_global, 
										 &y_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "x", im_global, 
										 &x_dimid));
		checkPnetcdfErrors(ncmpi_def_dim(ncid, "nb", nb, 
										 &nb_dimid));

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//1-D variable definations
		/*
		void def_var_pnetcdf(const int ncid, const char *name, 
							 const int nvdims, const int *vdims,
							 int *varid, const char *long_name,
							 const char *units){
		*/
		vdims[0] = time_dimid;
		strcpy(time_start_output, "days_since");
		strcat(time_start_output, my_trim(time_start, 26));
		def_var_pnetcdf(ncid, "time", 1, vdims, &time_varid,
						"time", time_start_output, " ", 0);

		vdims[0] = z_dimid;

		//////////////////////////////////////////////////////////////////
		def_var_pnetcdf(ncid, "z", 1, vdims, &z_varid,
						"sigma of cell face", "sigma_level", " ", 0);
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, z_varid, 
											  "standard_name",
											  22, "ocean_sigma_coordinate"));		
		checkPnetcdfErrors(ncmpi_put_att_text(ncid, z_varid, 
											  "formula_terms",
											  26, "sigma: z eta:elb depth:h"));		
		//////////////////////////////////////////////////////////////////
		def_var_pnetcdf(ncid, "zz", 1, vdims, &zz_varid, 
						"sigma of cell centre", "sigma_level", " ", 0);

		checkPnetcdfErrors(ncmpi_put_att_text(ncid, zz_varid, 
											  "standard_name",
											  22, "ocean_sigma_coordinate"));

		checkPnetcdfErrors(ncmpi_put_att_text(ncid, zz_varid, 
											  "formula_terms",
											  27, "sigma:z eta:elb depth:h"));

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//2-D variable definations
		vdims[1] = x_dimid;
		vdims[0] = y_dimid;
		def_var_pnetcdf(ncid, "dx", 2, vdims, &dx_varid,
						"grid increment in x", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "dy", 2, vdims, &dy_varid,
						"grid increment in y", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "east_u", 2, vdims, &east_u_varid,
						"easting of u-points", "degree",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "east_v", 2, vdims, &east_v_varid,
						"easting of v-points", "degree",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "east_e", 2, vdims, &east_e_varid,
						"easting of elevation points", "degree",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "east_c", 2, vdims, &east_c_varid,
						"easting of cell cormers", "degree",
						"east_c north_c", 1);
		def_var_pnetcdf(ncid, "north_u", 2, vdims, &north_u_varid,
						"northing of u-points", "degree",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "north_v", 2, vdims, &north_v_varid,
						"northing of v-points", "degree",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "north_e", 2, vdims, &north_e_varid,
						"northing of elevation points", "degree",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "north_c", 2, vdims, &north_c_varid,
						"northing of cell corners", "degree",
						"east_c north_c", 1);
		def_var_pnetcdf(ncid, "rot", 2, vdims, &rot_varid,
						"Rotation angle of x-axis wrt. east", "degree",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "h", 2, vdims, &h_varid,
						"undisturbed water depth", "metre",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "fsm", 2, vdims, &fsm_varid,
						"free surface mask", "dimensionless",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "dum", 2, vdims, &dum_varid,
						"u-velocity mask", "dimensionless",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "dvm", 2, vdims, &dvm_varid,
						"v-velocity mask", "dimensionless",
						"east_v north_v", 1);

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//3-D variable definations
		vdims[2] = x_dimid;
		vdims[1] = y_dimid;
		vdims[0] = time_dimid;

		def_var_pnetcdf(ncid, "uab", 3, vdims, &uab_varid,
						"depth-averaged u", "metre/sec",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "vab", 3, vdims, &vab_varid,
						"depth-averaged v", "metre/sec",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "elb", 3, vdims, &elb_varid,
						"surface elevation", "metre",
						"east_e north_e", 1);
		//xsz_check
		def_var_pnetcdf(ncid, "et", 3, vdims, &et_varid,
						"surface elevation", "metre",
						"east_e north_e", 1);

		def_var_pnetcdf(ncid, "wusurf", 3, vdims, &wusurf_varid,
						"x-momentum flux", "metre^2/sec^2",
						"east_u north_u", 1);
		def_var_pnetcdf(ncid, "wvsurf", 3, vdims, &wvsurf_varid,
						"y-momentum flux", "metre^2/sec^2",
						"east_v north_v", 1);
		def_var_pnetcdf(ncid, "wtsurf", 3, vdims, &wtsurf_varid,
						"temperature flux", "deg m/s",
						"east_e north_e", 1);
		def_var_pnetcdf(ncid, "wssurf", 3, vdims, &wssurf_varid,
						"salinity flux", "psu m/s",
						"east_e north_e", 1);

		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//4-D variable definations
		vdims[3] = x_dimid;
		vdims[2] = y_dimid;
		vdims[1] = z_dimid;
		vdims[0] = time_dimid;

		def_var_pnetcdf(ncid, "u", 4, vdims, &u_varid,
						"x-velocity", "metre/sec",
						"east_u north_u zz", 1);
		def_var_pnetcdf(ncid, "v", 4, vdims, &v_varid,
						"y-velocity", "metre/sec",
						"east_v north_v zz", 1);
		def_var_pnetcdf(ncid, "w", 4, vdims, &w_varid,
						"z-velocity", "metre/sec",
						"east_e north_e z", 1);
		def_var_pnetcdf(ncid, "t", 4, vdims, &t_varid,
						"potential temperature", "K",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "s", 4, vdims, &s_varid,
						"salinity x rho / rhoref", "PSS",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "rho", 4, vdims, &rho_varid,
						"(density-1000)/rhoref", "dimensionless",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "kh", 4, vdims, &kh_varid,
						"vertical diffusivity", "metre2/sec",
						"east_e north_e zz", 1);
		def_var_pnetcdf(ncid, "km", 4, vdims, &km_varid,
						"vertical viscosity", "metre2/sec",
						"east_e north_e zz", 1);
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//5-D variable definations
		vdims[4] = x_dimid;
		vdims[3] = y_dimid;
		vdims[2] = z_dimid;
		vdims[1] = nb_dimid;
		vdims[0] = time_dimid;

		def_var_pnetcdf(ncid, "tr", 5, vdims, &tr_varid,
						"tracer", "dimensionless",
						"east_u north_u zz nb", 1);

		checkPnetcdfErrors(ncmpi_enddef(ncid));

		///////////////////////////////////////////////////////////
		//1-D output	
		start[0] = 0;
		count[0] = kb;
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, z_varid, 
													start, count, z));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, zz_varid, 
													start, count, zz));
		///////////////////////////////////////////////////////////
		//2-D output	
		start[1] = i_global[0]-1;
		start[0] = j_global[0]-1;
		count[1] = im;
		count[0] = jm;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dx_varid,
											start, count, (float*)dx));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dy_varid,
											start, count, (float*)dy));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_u_varid,
											start, count, (float*)east_u));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_v_varid,
											start, count, (float*)east_v));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_e_varid,
											start, count, (float*)east_e));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, east_c_varid,
											start, count, (float*)east_c));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_u_varid,
											start, count, (float*)north_u));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_v_varid,
											start, count, (float*)north_v));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_e_varid,
											start, count, (float*)north_e));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, north_c_varid,
											start, count, (float*)north_c));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, rot_varid,
											start, count, (float*)rot));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, h_varid,
											start, count, (float*)h));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, fsm_varid,
											start, count, (float*)fsm));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dum_varid,
											start, count, (float*)dum));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, dvm_varid,
											start, count, (float*)dvm));

		///////////////////////////////////////////////////////////
		//3-D output	
		start[2] = i_global[0]-1;
		start[1] = j_global[0]-1;
		start[0] = 0;

		count[2] = im;
		count[1] = jm;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, uab_varid,
											start, count, (float*)uab));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, vab_varid,
											start, count, (float*)vab));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, elb_varid,
											start, count, (float*)elb));

		//xsz_check
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, et_varid,
											start, count, (float*)et));

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wusurf_varid,
											start, count, (float*)wusurf));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wvsurf_varid,
											start, count, (float*)wvsurf));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wtsurf_varid,
											start, count, (float*)wtsurf));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wssurf_varid,
											start, count, (float*)wssurf));

		///////////////////////////////////////////////////////////
		//4-D output	
		start[3] = i_global[0]-1;
		start[2] = j_global[0]-1;
		start[1] = 0;
		start[0] = 0;

		count[3] = im;
		count[2] = jm;
		count[1] = kb;
		count[0] = 1;

		printf("In write_output_pnetcdf, line:%d, start[2]=%d, start[3]=%d\n",
				__LINE__, start[2], start[3]);

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, u_varid,
											start, count, (float*)u));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, v_varid,
											start, count, (float*)v));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, w_varid,
											start, count, (float*)w));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, t_varid,
											start, count, (float*)t));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, s_varid,
											start, count, (float*)s));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, rho_varid,
											start, count, (float*)rho));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, kh_varid,
											start, count, (float*)kh));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, km_varid,
											start, count, (float*)km));

		///////////////////////////////////////////////////////////
		//5-D output	
		start[4] = i_global[0]-1;
		start[3] = j_global[0]-1;
		start[2] = 0;
		start[1] = 0;
		start[0] = 0;

		count[4] = im;
		count[3] = jm;
		count[2] = kb;
		count[1] = nb;
		count[1] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, tr_varid,
											start, count, (float*)tr));
		

	
	}else{
		if (my_task == master_task){
			printf("\n\n Openning Netcdf file: %s, num_out:%d\n\n", 
					netcdf_output_file, num_out);	
		}

		checkPnetcdfErrors(ncmpi_open(pom_comm, netcdf_output_file,
									  NC_WRITE, MPI_INFO_NULL, &ncid));

		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "time", &time_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "uab",  &uab_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "vab",  &vab_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "elb",  &elb_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wusurf", &wusurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wvsurf", &wvsurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wtsurf", &wtsurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "wssurf", &wssurf_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "u", &u_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "v", &v_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "w", &w_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "t", &t_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "s", &s_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "rho", &rho_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "kh", &kh_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "km", &km_varid));
		checkPnetcdfErrors(ncmpi_inq_varid(ncid, "tr", &tr_varid));

		///////////////////////////////////////////////////////////
		//1-D output	
		start[0] = num_out;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, time_varid, 
										start, count, (float*)&model_time));

		///////////////////////////////////////////////////////////
		//3-D output	
		start[2] = i_global[0]-1;
		start[1] = j_global[0]-1;
		start[0] = num_out;

		count[2] = im;
		count[1] = jm;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, uab_varid, 
										start, count, (float*)uab_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, vab_varid, 
										start, count, (float*)vab_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, elb_varid, 
										start, count, (float*)elb_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wusurf_varid, 
										start, count, (float*)wusurf_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wvsurf_varid, 
										start, count, (float*)wvsurf_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wtsurf_varid, 
										start, count, (float*)wtsurf_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, wssurf_varid, 
										start, count, (float*)wssurf_mean));

		///////////////////////////////////////////////////////////
		//4-D output	
		start[3] = i_global[0]-1;
		start[2] = j_global[0]-1;
		start[1] = 0;
		start[0] = num_out;

		count[3] = im;
		count[2] = jm;
		count[1] = kb;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, u_varid, 
										start, count, (float*)u_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, v_varid, 
										start, count, (float*)v_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, w_varid, 
										start, count, (float*)w_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, t_varid, 
										start, count, (float*)t_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, s_varid, 
										start, count, (float*)s_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, rho_varid, 
										start, count, (float*)rho_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, kh_varid, 
										start, count, (float*)kh_mean));
		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, km_varid, 
										start, count, (float*)km_mean));

		///////////////////////////////////////////////////////////
		//5-D output	
		start[4] = i_global[0]-1;
		start[3] = j_global[0]-1;
		start[2] = 0;
		start[1] = 0;
		start[0] = num_out;

		count[4] = im;
		count[3] = jm;
		count[2] = kb;
		count[1] = nb;
		count[0] = 1;

		checkPnetcdfErrors(ncmpi_put_vara_float_all(ncid, tr_varid, 
										start, count, (float*)tr));
	}

	checkPnetcdfErrors(ncmpi_sync(ncid));
	checkPnetcdfErrors(ncmpi_close(ncid));
	return;


/*
	for (k = 0; k < kb; k++){
		if (my_task == master_task)
			printf("z[%d] = %f\n",k ,z[k]);
		f_z[k] = z[k];
		f_zz[k] = zz[k];
	}

	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_dx[j][i] = dx[j][i];	
			f_dy[j][i] = dy[j][i];	
			f_east_u[j][i] = east_u[j][i];	
			f_east_v[j][i] = east_v[j][i];	
			f_east_e[j][i] = east_e[j][i];	
			f_east_c[j][i] = east_c[j][i];	
			f_north_u[j][i] = north_u[j][i];	
			f_north_v[j][i] = north_v[j][i];	
			f_north_e[j][i] = north_e[j][i];	
			f_north_c[j][i] = north_c[j][i];	

			f_rot[j][i] = rot[j][i];	
			f_h[j][i] = h[j][i];	
			f_fsm[j][i] = fsm[j][i];	
			f_dum[j][i] = dum[j][i];	
			f_dvm[j][i] = dvm[j][i];	
		}
	}
*/
}
