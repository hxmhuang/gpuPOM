#ifndef CIO_PNETCDF_H
#define CIO_PNETCDF_H

#include"data.h"

void read_grid_pnetcdf_();


void read_initial_ts_pnetcdf(int k,
							 float temp[][j_size][i_size],
							 float salt[][j_size][i_size]);

void read_mean_ts_pnetcdf(float temp[][j_size][i_size],
						  float salt[][j_size][i_size]);

void read_clim_ts_pnetcdf(float temp[][j_size][i_size],
						  float salt[][j_size][i_size]);

void read_tide_east_pnetcdf(float ampe_out[][j_size],
							float phae_out[][j_size],
							float amue_out[][j_size],
							float phue_out[][j_size]);

void def_var_pnetcdf(const int ncid, const char *name, 
					 const int nvdims, const int *vdims,
					 int *varid, 
					 const char *long_name,
					 const char *units,
					 const char *coords,
					 int lcoords);

void *write_output_ptr(int num_out);

void write_output(int num_out);

void write_output_pnetcdf();
#endif
