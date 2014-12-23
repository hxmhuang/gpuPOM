#ifndef CPARALLEL_MPI_H
#define CPARALLEL_MPI_H

#include"data.h"

#ifdef __NVCC__
extern "C" {
#endif

void initialize_mpi();
/*
void initialize_mpi_(int *f_my_task, 
					 int *f_pom_comm, 
					 int *f_pom_comm_coarse,
					 int *f_master_task,
					 int *f_error_status);
*/

void distribute_mpi();

/*
void distribute_mpi_(int *f_im, int *f_imm1, int *f_imm2,
					int *f_jm, int *f_jmm1, int *f_jmm2,
					int *f_kbm1, int *f_kbm2,
					int *f_n_east, int *f_n_west,
					int *f_n_north, int *f_n_south,
					int *f_i_global, int *f_j_global);
*/

void distribute_mpi_coarse();
/*
void distribute_mpi_coarse_(int *f_im_coarse, 
							int *f_jm_coarse,
							int *f_i_global_coarse,
							int *f_j_global_coarse);
*/

void sum0i_mpi(int *work, int root);

void sum0f_mpi(float *work, int root);

void bcast0d_mpi(int *work, int from);

void finalize_mpi();


void exchange3d_mpi_bak(float work[][j_size][i_size], 
						 int nx, int ny, int zstart, int zend);

void exchange3d_mpi(float work[][j_size][i_size], 
						 int nx, int ny, int nz);

void exchange3d_mpi_tag(float work[][j_size][i_size], 
						int nx, int ny, int nz, int tag);

void exchange2d_mpi(float work[][i_size], 
						 int nx, int ny);

void exchange2d_mpi_tag(float work[][i_size], int nx, int ny, int tag);

void order3d_mpi_(float work2[][j_size][i_size], 
				 float work4[][j_size+1][i_size+1],
				 int nx, int ny, int nz);

void order2d_mpi(float work2[][i_size], 
				 float work4[][i_size+1], 
				 int nx, int ny);

void xperi2d_mpi(float work[][i_size],
				 int nx, int ny);

#ifdef __NVCC__
}
#endif

#endif
