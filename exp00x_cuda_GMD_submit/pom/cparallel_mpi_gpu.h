#ifndef CPARALLEL_MPI_GPU_H
#define CPARALLEL_MPI_GPU_H

/*
void initialize_mpi_gpu();

void distribute_mpi_gpu();

int sum0d_mpi_gpu(int work, int root);

void bcast0d_mpi_gpu(int *work, int from);

void finalize_mpi_gpu();


void exchange3d_mpi_bak(float work[][j_size][i_size], 
						 int nx, int ny, int zstart, int zend);

void exchange3d_mpi(float work[][j_size][i_size], 
						 int nx, int ny, int nz);


void exchange2d_mpi(float work[][i_size], 
					int nx, int ny);

void exchange2d_mpi_bak(float work[][i_size], 
						int nx, int ny);
*/
void exchange3d_mpi_gpu(float *d_work, int nx, int ny, int nz);

void exchange3d_cuda_aware_mpi(float *d_work, int nx, int ny, int nz);

void exchange2d_mpi_gpu(float *d_work, int nx, int ny);

void exchange2d_cuda_aware_mpi(float *d_work, int nx, int ny);

void xperi2d_mpi_gpu(float *d_work, int nx, int ny);

void xperi2d_cuda_aware_mpi(float *d_work, int nx, int ny);

void yperi2d_mpi_gpu(float *d_work, int nx, int ny);

void xperi3d_mpi_gpu(float *d_work, int nx, int ny, int nz);

void xperi3d_cuda_aware_mpi(float *d_work, int nx, int ny, int nz);

void yperi3d_mpi_gpu(float *d_work, int nx, int ny, int nz);

void exchange3d_cuda_ipc(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny, int nz);

void exchange2d_cuda_ipc(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny);

void send_east_cuda_ipc(float *d_send, 
						float *d_east_recv, 
						cudaStream_t &stream_in,
						int nx, int ny);

void xperi2d_cuda_ipc(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny);

void xperi3d_cuda_ipc(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny, int nz);

void exchange2d_cudaPeer(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 float *d_south_recv, 
						 float *d_north_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny);

void exchange2d_cudaPeerAsync(float *d_send, 
							  float *d_east_recv, 
						      float *d_west_recv,
							  float *d_south_recv, 
							  float *d_north_recv,
							  cudaStream_t &stream_in,
							  int nx, int ny);

void exchange3d_cudaPeer(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 float *d_south_recv, 
						 float *d_north_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny, int nz);

void exchange3d_cudaPeerAsync(float *d_send, 
						      float *d_east_recv, 
						      float *d_west_recv,
						      float *d_south_recv, 
						      float *d_north_recv,
						      cudaStream_t &stream_in,
						      int nx, int ny, int nz);

void exchange3d_cudaDHD(float *d_work, cudaStream_t &stream_in,
					int nx, int ny, int nz);
void exchange2d_cudaDHD(float *d_work, cudaStream_t &stream_in,
					int nx, int ny);

void xperi2d_cudaPeer(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny);

void xperi2d_cudaPeerAsync(float *d_work, 
					       float *d_east_most_recv,
					       float *d_west_most_recv,
					       cudaStream_t &stream_in,
					       int nx, int ny);

void xperi3d_cudaPeer(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny, int nz);

void xperi3d_cudaPeerAsync(float *d_work, 
					       float *d_east_most_recv,
					       float *d_west_most_recv,
					       cudaStream_t &stream_in,
					       int nx, int ny, int nz);

void exchange2d_cudaUVA(float *d_send, 
						float *d_east_recv, 
						float *d_west_recv,
						float *d_south_recv, 
						float *d_north_recv,
						cudaStream_t &stream_in,
						int nx, int ny);

void exchange2d_cudaUVAAsync(float *d_send, 
						     float *d_east_recv, 
						     float *d_west_recv,
						     float *d_south_recv, 
						     float *d_north_recv,
						     cudaStream_t &stream_in,
						     int nx, int ny);

void exchange3d_cudaUVA(float *d_send, 
						float *d_east_recv, 
						float *d_west_recv,
						float *d_south_recv,
						float *d_north_recv, 
						cudaStream_t &stream_in,
						int nx, int ny, int nz);

void exchange3d_cudaUVAAsync(float *d_send, 
						     float *d_east_recv, 
						     float *d_west_recv,
						     float *d_south_recv,
						     float *d_north_recv, 
						     cudaStream_t &stream_in,
						     int nx, int ny, int nz);

void xperi2d_cudaUVA(float *d_work, 
					 float *d_east_most_recv,
					 float *d_west_most_recv,
					 cudaStream_t &stream_in,
					 int nx, int ny);

void xperi2d_cudaUVAAsync(float *d_work, 
						  float *d_east_most_recv,
						  float *d_west_most_recv,
					      cudaStream_t &stream_in,
					      int nx, int ny);

void xperi3d_cudaUVA(float *d_work, 
					 float *d_east_most_recv,
					 float *d_west_most_recv,
					 cudaStream_t &stream_in,
					 int nx, int ny, int nz);

void xperi3d_cudaUVAAsync(float *d_work, 
					      float *d_east_most_recv,
					      float *d_west_most_recv,
					      cudaStream_t &stream_in,
					      int nx, int ny, int nz);
#endif
