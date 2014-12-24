#include<stdio.h>
#include<mpi.h>

#include"cparallel_mpi.h"
#include"timer_all.h"

/*
void initialize_mpi_(int *f_my_task, 
					 int *f_pom_comm, 
					 int *f_pom_comm_coarse,
					 int *f_master_task,
					 int *f_error_status){
*/

void initialize_mpi(){

	MPI_Init(NULL, NULL);
	int origin_size;
	MPI_Comm_size(MPI_COMM_WORLD, &origin_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &origin_task);
	int *ranks_gpu = (int*)malloc((origin_size/2)*sizeof(int));
	int *ranks_io = (int*)malloc((origin_size/2)*sizeof(int));

	int i;
	for (i = 0; i < origin_size/2; i++){
		ranks_gpu[i] = i;
		ranks_io[i] = i+origin_size/2;
	}

	MPI_Group origin_group, new_group;
	MPI_Comm_group(MPI_COMM_WORLD, &origin_group);

	if (origin_task < origin_size/2){
		MPI_Group_incl(origin_group, origin_size/2, ranks_gpu, &new_group);	
	}else{
		MPI_Group_incl(origin_group, origin_size/2, ranks_io, &new_group);
	}

	MPI_Comm_create(MPI_COMM_WORLD, new_group, &pom_comm);
	pom_comm_coarse = pom_comm;

	MPI_Comm_rank(pom_comm, &my_task);

	//int provided;
	//MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
	//MPI_Comm_rank(MPI_COMM_WORLD, &my_task);
	//printf("my_task = %d: provided = %d\n", my_task, provided);

	//pom_comm = MPI_COMM_WORLD;
	//pom_comm_coarse = MPI_COMM_WORLD;
	master_task = 0;
	error_status = 0;

	MPI_Barrier(MPI_COMM_WORLD);


	/*
	*f_my_task = my_task;
	*f_pom_comm = pom_comm;
	*f_pom_comm_coarse = pom_comm_coarse;
	*f_master_task = master_task;
	*f_error_status = error_status;
	*/

}


/*
void distribute_mpi_(int *f_im, int *f_imm1, int *f_imm2,
					int *f_jm, int *f_jmm1, int *f_jmm2,
					int *f_kbm1, int *f_kbm2,
					int *f_n_east, int *f_n_west,
					int *f_n_north, int *f_n_south,
					int *f_i_global, int *f_j_global){
*/

void distribute_mpi(){

	int i, j, ierr, nproc, nproc_x, nproc_y;

	/*
	n_proc = *f_n_proc;
	im_global = *f_im_global;
	jm_global = *f_jm_global;
	im_local= *f_im_local;
	jm_local= *f_jm_local;
	kb = *f_kb;
	*/
	
	
	
//! determine the number of processors
	MPI_Comm_size(pom_comm, &nproc);

//! check number of processors
	if (nproc != n_proc){
		error_status = 1;	
		if (my_task == master_task){
			printf("Incompatible number of processors\nPom terminated with error\n");	
		}
		finalize_mpi();
		exit(1);
	}

//! determine the number of processors in x
	if ((im_global-2)%(im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x	= (im_global-2)/(im_local-2) + 1;
	}

//! determine the number of processors in y
	if ((jm_global-2)%(jm_local-2) == 0){
		nproc_y = (jm_global-2)/(jm_local-2);
	}else{
		nproc_y = (jm_global-2)/(jm_local-2) + 1;
	}

//! check local size
	if (nproc_x*nproc_y > n_proc){
		error_status = 1;	
		if (my_task == master_task){
			printf("im_local or jm_local is too low\nPom terminated with error\n");	
		}
		finalize_mpi();
		exit(1);
	}
	
//! detemine global and local indices
	im = im_local;
	for (i = 0; i < im_local; i++){
		i_global[i] = 0;	
	}

	for (i = 0; i < im; i++){
		i_global[i] = (i+1)	+ (my_task%nproc_x)*(im-2);
		if (i_global[i] > im_global){
			im = i-1;
			i_global[i] = 0;
			continue;
		}
	}

	imm1 = im-1;
	imm2 = im-2;

	jm = jm_local;
	for (j = 0; j < jm_local; j++){
		j_global[j] = 0;	
	}

	for (j = 0; j < jm; j++){
		j_global[j] = (j+1)+(my_task/nproc_x)*(jm-2);
		if (j_global[j] > jm_global){
			jm = j-1;	
			j_global[j] = 0;
			continue;
		}
	}

	jmm1 = jm-1;
	jmm2 = jm-2;
	kbm1 = kb-1;
	kbm2 = kb-2;

//! determine the neighbors (tasks)
	n_east = my_task+1;
	n_west = my_task-1;
	n_north = my_task+nproc_x;
	n_south = my_task-nproc_x;

	if ((n_east%nproc_x) == 0)
		n_east = -1;
	if ((n_west+1)%nproc_x == 0)
		n_west = -1;
	if (n_north/nproc_x == nproc_y)
		n_north = -1;
	if ((n_south+nproc_x)/nproc_x == 0)
		n_south = -1;
	
	/*
	*f_im = im;
	*f_imm1 = imm1;
	*f_imm2 = imm2;

	*f_jm = jm;
	*f_jmm1 = jmm1;
	*f_jmm2 = jmm2;

	*f_kbm1 = kbm1;
	*f_kbm2 = kbm2;

	*f_n_east = n_east;
	*f_n_west = n_west;
	*f_n_north = n_north;
	*f_n_south = n_south;

	for (i = 0; i < im; i++){
		f_i_global[i] = i_global[i];
	}
	for (j = 0; j < jm; j++){
		f_j_global[j] = j_global[j];	
	}
	*/
	
	return;
}


/*
void distribute_mpi_coarse_(int *f_im_coarse, 
							int *f_jm_coarse,
							int *f_i_global_coarse,
							int *f_j_global_coarse){
*/
void distribute_mpi_coarse(){

    int i,j,ierr,nproc,nproc_x,nproc_y;
//! determine the number of processors
	MPI_Comm_size(pom_comm_coarse, &nproc);

	if (nproc != n_proc){
		error_status = 1;	
		if (my_task == master_task){
			printf("Incompatible number of processors\nPom terminated with error\n");	
		}
		finalize_mpi();
		exit(1);
	}

//! determine the number of processors in x
	if ((im_global_coarse-2)%(im_local_coarse-2) == 0){
		nproc_x = (im_global_coarse-2)/(im_local_coarse-2);
	}else{
		nproc_x	= (im_global_coarse-2)/(im_local_coarse-2) + 1;
	}

//! determine the number of processors in y
	if ((jm_global_coarse-2)%(jm_local_coarse-2) == 0){
		nproc_y = (jm_global_coarse-2)/(jm_local_coarse-2);
	}else{
		nproc_y = (jm_global_coarse-2)/(jm_local_coarse-2) + 1;
	}

//! check local size
	if (nproc_x*nproc_y > n_proc){
		error_status = 1;	
		if (my_task == master_task){
			printf("im_local or jm_local is too low\nPom terminated with error\n");	
		}
		finalize_mpi();
		exit(1);
	}
	
//! detemine global and local indices
	im_coarse = im_local_coarse;
	for (i = 0; i < im_local_coarse; i++){
		i_global_coarse[i] = 0;	
	}

	for (i = 0; i < im_local_coarse; i++){
		i_global_coarse[i] = (i+1)	+ (my_task%nproc_x)*(im_coarse-2);
		if (i_global_coarse[i] > im_global_coarse){
			im_coarse = i-1;
			i_global_coarse[i] = 0;
			continue;
		}
	}

	jm_coarse = jm_local_coarse;
	for (j = 0; j < jm_local_coarse; j++){
		j_global_coarse[j] = 0;	
	}

	for (j = 0; j < jm_local_coarse; j++){
		j_global_coarse[j] = (j+1)+(my_task/nproc_x)*(jm_coarse-2);
		if (j_global_coarse[j] > jm_global_coarse){
			jm_coarse = j-1;	
			j_global_coarse[j] = 0;
			continue;
		}
	}

	/*
	*f_im_coarse = im_coarse;
	*f_jm_coarse = jm_coarse;

	for (i = 0; i < im_local_coarse; i++){
		f_i_global_coarse[i] = i_global_coarse[i];
	}
	for (j = 0; j < jm_local_coarse; j++){
		f_j_global_coarse[j] = j_global_coarse[j];	
	}
	*/

	return; 
}

void sum0i_mpi(int *work, int root){
	int work_recv = 0;
	MPI_Reduce(work, &work_recv, 1, MPI_INT, MPI_SUM, root, pom_comm);
	*work = work_recv;
	return;
}


void sum0f_mpi(float *work, int root){
	float work_recv = 0;
	MPI_Reduce(work, &work_recv, 1, MPI_FLOAT, MPI_SUM, root, pom_comm);
	*work = work_recv;
	return;
}


void bcast0d_mpi(int *work, int from){
	MPI_Bcast(work, 1, MPI_INT, from, pom_comm);	
	return;
}

void bcastf_mpi(float *work, int elts, int from){
	MPI_Bcast(work, elts, MPI_FLOAT, from, pom_comm);
	return;
}

void finalize_mpi(){
	MPI_Finalize();	
}

/*
void exchange3d_mpi_xsz(float *work, int nx, int ny, int nz){

	int i, j, k;
	MPI_Status status;
	float *send_east = (float*)malloc(ny*nz*sizeof(float));
	float *recv_west= (float*)malloc(ny*nz*sizeof(float));
	float *send_west = (float*)malloc(ny*nz*sizeof(float));
	float *recv_east= (float*)malloc(ny*nz*sizeof(float));
	float *send_north= (float*)malloc(ny*nz*sizeof(float));
	float *recv_south= (float*)malloc(ny*nz*sizeof(float));
	float *send_south= (float*)malloc(ny*nz*sizeof(float));
	float *recv_north= (float*)malloc(ny*nz*sizeof(float));
	
	//printf("in exchange3d: I come to here0\n");
	if (n_east != -1){
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;
				send_east[i] = work[k*ny*nx+j*nx+nx-2];
			}
		}
		MPI_Send(send_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm);
	}

	if (n_west != -1){
		MPI_Recv(recv_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &status);	
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work[k*ny*nx+j*nx] = recv_west[i];
			}
		}
	}

// send ghost cell data to the west
	if (n_west != -1){
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i =j+k*ny;
				send_west[i] = work[k*ny*nx+j*nx+1];
			}
		}
		MPI_Send(send_west, ny*nz, MPI_FLOAT, n_west, my_task, pom_comm);
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		MPI_Recv(recv_east, ny*nz, MPI_FLOAT, n_east, n_east, pom_comm, &status);
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work[k*ny*nx+j*nx+nx-1] = recv_east[i];
			}
		}
	}

// send ghost cell data to the north
	if (n_north != -1){
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				send_north[j] = work[k*ny*nx+(ny-2)*nx+i];
			}
		}
		MPI_Send(send_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &status);
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				work[k*ny*nx+i] = recv_south[j];
			}
		}
	}

// send ghost cell data to the south
	if (n_south != -1){
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				send_south[j] = work[k*ny*nx+1*nx+i];
			}
		}
		MPI_Send(send_south, nx*nz, MPI_FLOAT, n_south, my_task, pom_comm);
	}

// recieve ghost cell data from the north
	if (n_north != -1){
		MPI_Recv(recv_north, nx*nz, MPI_FLOAT, n_north, n_north, pom_comm, &status);
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				work[k*ny*nx+(ny-1)*nx+i] = recv_north[j];
			}
		}
	}

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);

	return;

}
*/

//void exchange3d_mpi_xsz_(float work[][j_size][i_size], int *nx_in, int *ny_in, int *zstart_in, int *zend_in){
void exchange3d_mpi_bak(float work[][j_size][i_size], int nx, int ny, int zstart, int zend){

	int i, j, k;
	MPI_Status status;

	
	/*
	int nx = *nx_in;
	int ny = *ny_in;
	int zstart = *zstart_in;
	int zend = *zend_in;
	*/
	
	
	float *send_east = (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *recv_west= (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *send_west = (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *recv_east= (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *send_north= (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *recv_south= (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *send_south= (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	float *recv_north= (float*)malloc(ny*(zend-zstart+1)*sizeof(float));
	
	//printf("in exchange3d: I come to here0\n");
	if (n_east != -1){
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+(k-zstart)*ny;
				send_east[i] = work[k][j][nx-2];
			}
		}
		MPI_Send(send_east, ny*(zend-zstart+1), MPI_FLOAT, n_east, my_task, pom_comm);
	}

	if (n_west != -1){
		MPI_Recv(recv_west, ny*(zend-zstart+1), MPI_FLOAT, n_west, n_west, pom_comm, &status);	
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+(k-zstart)*ny;	
				work[k][j][0] = recv_west[i];
			}
		}
	}

// send ghost cell data to the west
	if (n_west != -1){
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i =j+(k-zstart)*ny;
				send_west[i] = work[k][j][1];
			}
		}
		MPI_Send(send_west, ny*(zend-zstart+1), MPI_FLOAT, n_west, my_task, pom_comm);
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		MPI_Recv(recv_east, ny*(zend-zstart+1), MPI_FLOAT, n_east, n_east, pom_comm, &status);
		for (k = zstart; k <=zend; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+(k-zstart)*ny;	
				work[k][j][nx-1] = recv_east[i];
			}
		}
	}

// send ghost cell data to the north
	if (n_north != -1){
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+(k-zstart)*nx;	
				send_north[j] = work[k][ny-2][i];
			}
		}
		MPI_Send(send_north, nx*(zend-zstart+1), MPI_FLOAT, n_north, my_task, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx*(zend-zstart+1), MPI_FLOAT, n_south, n_south, pom_comm, &status);
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+(k-zstart)*nx;
				work[k][0][i] = recv_south[j];
			}
		}
	}

// send ghost cell data to the south
	if (n_south != -1){
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+(k-zstart)*nx;
				send_south[j] = work[k][1][i];
			}
		}
		MPI_Send(send_south, nx*(zend-zstart+1), MPI_FLOAT, n_south, my_task, pom_comm);
	}

// recieve ghost cell data from the north
	if (n_north != -1){
		MPI_Recv(recv_north, nx*(zend-zstart+1), MPI_FLOAT, n_north, n_north, pom_comm, &status);
		for (k = zstart; k <= zend; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+(k-zstart)*nx;	
				work[k][ny-1][i] = recv_north[j];
			}
		}
	}

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);

	return;

}

void exchange3d_mpi(float work[][j_size][i_size], int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_mpi, 
				   time_end_exchange3d_mpi;
	timer_now(&time_start_exchange3d_mpi);
#endif
	int i, j, k;
	MPI_Status status;

	
	float *send_east = (float*)malloc(ny*nz*sizeof(float));
	float *recv_west= (float*)malloc(ny*nz*sizeof(float));
	float *send_west = (float*)malloc(ny*nz*sizeof(float));
	float *recv_east= (float*)malloc(ny*nz*sizeof(float));
	float *send_north= (float*)malloc(nx*nz*sizeof(float));
	float *recv_south= (float*)malloc(nx*nz*sizeof(float));
	float *send_south= (float*)malloc(nx*nz*sizeof(float));
	float *recv_north= (float*)malloc(nx*nz*sizeof(float));
	
	//printf("in exchange3d: I come to here0\n");
	if (n_east != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+k*ny;
				send_east[i] = work[k][j][nx-2];
			}
		}
		MPI_Send(send_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm);
	}

	if (n_west != -1){
		MPI_Recv(recv_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &status);	
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work[k][j][0] = recv_west[i];
			}
		}
	}

// send ghost cell data to the west
	if (n_west != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i =j+k*ny;
				send_west[i] = work[k][j][1];
			}
		}
		MPI_Send(send_west, ny*nz, MPI_FLOAT, n_west, my_task, pom_comm);
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		MPI_Recv(recv_east, ny*nz, MPI_FLOAT, n_east, n_east, pom_comm, &status);
		for (k = 0; k < nz ; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work[k][j][nx-1] = recv_east[i];
			}
		}
	}

// send ghost cell data to the north
	if (n_north != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				send_north[j] = work[k][ny-2][i];
			}
		}
		MPI_Send(send_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &status);
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				work[k][0][i] = recv_south[j];
			}
		}
	}

// send ghost cell data to the south
	if (n_south != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				send_south[j] = work[k][1][i];
			}
		}
		MPI_Send(send_south, nx*nz, MPI_FLOAT, n_south, my_task, pom_comm);
	}

// recieve ghost cell data from the north
	if (n_north != -1){
		MPI_Recv(recv_north, nx*nz, MPI_FLOAT, n_north, n_north, pom_comm, &status);
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				work[k][ny-1][i] = recv_north[j];
			}
		}
	}

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);

#ifndef TIME_DISABLE
	timer_now(&time_end_exchange3d_mpi);
	exchange3d_mpi_time += time_consumed(&time_start_exchange3d_mpi,
									     &time_end_exchange3d_mpi);
#endif

	return;

}

//!_______________________________________________________________________
//!fhx:Toni:npg
void order3d_mpi(float work2[][j_size][i_size], 
				 float work4[][j_size+1][i_size+1],
				 int nx, int ny, int nz){

//! convert a 2nd order 3d matrix to special 4th order 3d matrix
	int i, j, k;
	MPI_Status status;

	
	float *send_east = (float*)malloc(ny*nz*sizeof(float));
	float *recv_west= (float*)malloc(ny*nz*sizeof(float));
	float *send_north= (float*)malloc(nx*nz*sizeof(float));
	float *recv_south= (float*)malloc(nx*nz*sizeof(float));
	
	//printf("in exchange3d: I come to here0\n");
	for (i = 0; i < nx; i++){
		for (j = 0; j < ny; j++){
			for (k = 0; k < nz; k++){
				work4[k][j+1][i+1] = work2[k][j][i];
			}
		}
	}

//! send ghost cell data to the east

	if (n_east != -1){
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;
				send_east[i] = work2[k][j][nx-3];
			}
		}
		MPI_Send(send_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm);
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		MPI_Recv(recv_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &status);	
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work4[k][j+1][0] = recv_west[i];
			}
		}
	}

// send ghost cell data to the north
	if (n_north != -1){
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				send_north[j] = work2[k][ny-3][i];
			}
		}
		MPI_Send(send_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &status);
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				work4[k][0][i+1] = recv_south[j];
			}
		}
	}

	free(send_east);
	free(recv_west);
	free(send_north);
	free(recv_south);

	return;
}


/*
      subroutine exchange2d_mpi(work,nx,ny)
! exchange ghost cells around 2d local grids
! one band at a time
      implicit none
      include 'mpif.h'
      include 'pom.h'
      integer nx,ny
      real work(nx,ny)
      integer i,j,k
      integer ierr
      integer istatus(mpi_status_size)
      real send_east(ny),recv_west(ny)
      real send_west(ny),recv_east(ny)
      real send_north(nx),recv_south(nx)
      real send_south(nx),recv_north(nx)
      integer Work_vector

      double precision exchange2d_mpi_time_start_xsz
      double precision exchange2d_mpi_time_end_xsz
    
      call call_time_start(exchange2d_mpi_time_start_xsz)
*/
void exchange2d_mpi(float work[][i_size], int nx, int ny){
//void exchange2d_mpi_xsz_(float work[][i_size], int *f_nx, int *f_ny){

	/*
	int nx = *f_nx;
	int ny = *f_ny;
	*/

#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_mpi, 
				   time_end_exchange2d_mpi;
	timer_now(&time_start_exchange2d_mpi);
#endif

	float *send_east = (float*)malloc(ny*sizeof(float));
	float *recv_west = (float*)malloc(ny*sizeof(float));
	float *send_west = (float*)malloc(ny*sizeof(float));
	float *recv_east = (float*)malloc(ny*sizeof(float));

	float *send_north = (float*)malloc(nx*sizeof(float));
	float *recv_south = (float*)malloc(nx*sizeof(float));
	float *send_south = (float*)malloc(nx*sizeof(float));
	float *recv_north = (float*)malloc(nx*sizeof(float));

	int i,j;
	MPI_Status status;
//! send ghost cell data to the east
/*
      if(n_east.ne.-1) then
        do j=1,ny
          send_east(j)=work(nx-1,j)
        end do
        call mpi_send(send_east,ny,mpi_real,n_east,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(nx-1, 1), 1, Work_vector, n_east, my_task, 
!     $                pom_comm, ierr)
      end if
*/
	if (n_east != -1){
		for (j = 0; j < ny; j++){
			send_east[j] = work[j][nx-2];	
		}
		MPI_Send(send_east, ny, MPI_FLOAT, n_east, my_task, pom_comm);
	}

//! recieve ghost cell data from the west
/*
      if(n_west.ne.-1) then
        call mpi_recv(recv_west,ny,mpi_real,n_west,n_west,
     $                pom_comm,istatus,ierr)
        do j=1,ny
          work(1,j)=recv_west(j)
        end do
!        call MPI_RECV(work(1, 1), 1, Work_vector, n_west, n_west, 
!     $                pom_comm, istatus, ierr)
      end if
*/
	if (n_west != -1){
		MPI_Recv(recv_west, ny, MPI_FLOAT, n_west, n_west, pom_comm, &status);	
		for (j = 0; j < ny; j++){
			work[j][0] = recv_west[j];
		}
	}

//! send ghost cell data to the west
/*
      if(n_west.ne.-1) then
        do j=1,ny
          send_west(j)=work(2,j)
        end do
        call mpi_send(send_west,ny,mpi_real,n_west,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(2,1), 1, Work_vector, n_west, my_task,
!     $                pom_comm,ierr)
      end if
*/

	if (n_west != -1){
		for (j = 0; j < ny; j++){
			send_west[j] = work[j][1];
		}
		MPI_Send(send_west, ny, MPI_FLOAT, n_west, my_task, pom_comm);
	}

//! recieve ghost cell data from the east
/*
      if(n_east.ne.-1) then
        call mpi_recv(recv_east,ny,mpi_real,n_east,n_east,
     $                pom_comm,istatus,ierr)
        do j=1,ny
          work(nx,j)=recv_east(j)
        end do
!        call MPI_RECV(work(nx, 1), 1, Work_vector, n_east, n_east,
!     $                pom_comm,istatus,ierr)
      end if
*/
	if (n_east != -1){
		MPI_Recv(recv_east, ny, MPI_FLOAT, n_east, n_east, pom_comm, &status);	
		for (j = 0; j < ny; j++){
			work[j][nx-1] = recv_east[j];	
		}
	}

//! send ghost cell data to the north
/*
      if(n_north.ne.-1) then
        do i=1,nx
          send_north(i)=work(i,ny-1)
        end do
        call mpi_send(send_north,nx,mpi_real,n_north,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(1, ny-1),nx,mpi_real,n_north,my_task,
!     $                pom_comm,ierr)
      end if
*/
	if (n_north != -1){
		for (i = 0; i < nx; i++){
			send_north[i] = work[ny-2][i];	
		}
		MPI_Send(send_north, nx, MPI_FLOAT, n_north, my_task, pom_comm);
	}

//! recieve ghost cell data from the south
/*
      if(n_south.ne.-1) then
        call mpi_recv(recv_south,nx,mpi_real,n_south,n_south,
     $                pom_comm,istatus,ierr)
        do i=1,nx
          work(i,1)=recv_south(i)
        end do
!        call MPI_RECV(work(1, 1),nx,mpi_real,n_south,n_south,
!     $                pom_comm,istatus,ierr)
      

      end if
*/
	if (n_south != -1){
		MPI_Recv(recv_south, nx, MPI_FLOAT, n_south, n_south, pom_comm, &status);	
		for (i = 0; i < nx; i++){
			work[0][i] = recv_south[i];	
		}
	}

//! send ghost cell data to the south
/*
      if(n_south.ne.-1) then
        do i=1,nx
          send_south(i)=work(i,2)
        end do
        call mpi_send(send_south,nx,mpi_real,n_south,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(1, 2),nx,mpi_real,n_south,my_task,
!     $                pom_comm,ierr)
      end if
*/
	if (n_south != -1){
		for (i = 0; i < nx; i++){
			send_south[i] = work[1][i];	
		}
		MPI_Send(send_south, nx, MPI_FLOAT, n_south, my_task, pom_comm);
	}

//! recieve ghost cell data from the north
/*
      if(n_north.ne.-1) then
        call mpi_recv(recv_north,nx,mpi_real,n_north,n_north,
     $                pom_comm,istatus,ierr)
        do i=1,nx
          work(i,ny)=recv_north(i)
        end do
!        call MPI_RECV(work(1, ny),nx,mpi_real,n_north,n_north,
!     $                pom_comm,istatus,ierr)
      end if
*/
	if (n_north != -1){
		MPI_Recv(recv_north, nx, MPI_FLOAT, n_north, n_north, pom_comm, &status);
		for (i = 0; i < nx; i++){
			work[ny-1][i] = recv_north[i];	
		}
	}

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);
#ifndef TIME_DISABLE
	timer_now(&time_end_exchange2d_mpi);
	exchange2d_mpi_time += time_consumed(&time_start_exchange2d_mpi,
									     &time_end_exchange2d_mpi);
#endif

	return;
}


//!_______________________________________________________________________
//!fhx:Toni:npg

void order2d_mpi(float work2[][i_size], 
				 float work4[][i_size+1], 
				 int nx, int ny){
	int i, j, k;	

	float *send_east = (float*)malloc(ny*sizeof(float));
	float *recv_west = (float*)malloc(ny*sizeof(float));

	float *send_north = (float*)malloc(nx*sizeof(float));
	float *recv_south = (float*)malloc(nx*sizeof(float));

	for (i = 0; i < nx; i++){
		for (j = 0; j < ny; j++){
			work4[j+1][i+1] = work2[j][i];	
		}
	}

	MPI_Status status;
//! send ghost cell data to the east
	if (n_east != -1){
		for (j = 0; j < ny; j++){
			send_east[j] = work2[j][nx-3];	
		}
		MPI_Send(send_east, ny, MPI_FLOAT, n_east, my_task, pom_comm);
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		MPI_Recv(recv_west, ny, MPI_FLOAT, n_west, n_west, pom_comm, &status);	
		for (j = 0; j < ny; j++){
			work4[j+1][0] = recv_west[j];
		}
	}

//! send ghost cell data to the north
	if (n_north != -1){
		for (i = 0; i < nx; i++){
			send_north[i] = work2[ny-3][i];	
		}
		MPI_Send(send_north, nx, MPI_FLOAT, n_north, my_task, pom_comm);
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx, MPI_FLOAT, n_south, n_south, pom_comm, &status);	
		for (i = 0; i < nx; i++){
			work4[0][i+1] = recv_south[i];	
		}
	}

	free(send_east);
	free(recv_west);
	free(send_north);
	free(recv_south);

	return;
}


/*
      subroutine xperi2d_mpi(wrk,nx,ny)
! doing periodic bc in x
! pass from east to west and also pass from west to east
      implicit none
      include 'mpif.h'
      include 'pom.h'
      integer nx,ny
      real wrk(nx,ny)
      integer i,j,k
      integer ierr,nproc_x,nproc_y
      integer dest_task,sour_task
      integer istatus(mpi_status_size)
      real sendbuf(ny),recvbuf(ny)
*/
void xperi2d_mpi(float work[][i_size],
				 int nx, int ny){
	int i, j, k;
	int nproc_x, nproc_y;
	int dest_task, sour_task;
	float *sendbuf = (float*)malloc(ny*sizeof(float));
	float *recvbuf = (float*)malloc(ny*sizeof(float));

	MPI_Status status;
/*
! determine the number of processors in x
      if(mod(im_global-2,im_local-2).eq.0) then
        nproc_x=(im_global-2)/(im_local-2)
      else
        nproc_x=(im_global-2)/(im_local-2) + 1
      end if
*/

	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

/*
      if (nproc_x.eq.1) then
        do j=1,ny
        wrk(nx,j)=wrk(3,j); wrk(1,j)=wrk(nx-2,j); wrk(2,j)=wrk(nx-1,j)
        enddo
      else
!
C  !The most east sudomains    
      if(n_east.eq.-1) then
        dest_task=my_task-nproc_x+1
        sour_task=my_task-nproc_x+1

       ! first time to send
         do j=1,ny
           sendbuf(j)=wrk(nx-2,j)
         end do
         call mpi_send(sendbuf,ny,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)


       !first time to recieve
         call mpi_recv(recvbuf,ny,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do j=1,ny
          wrk(nx,j)=recvbuf(j)
         end do

       ! second time to send
         do j=1,ny
           sendbuf(j)=wrk(nx-1,j)
         end do
         call mpi_send(sendbuf,ny,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

      endif !if(n_east.eq.-1)

C  !The most west sudomains    
      if(n_west.eq.-1) then
        sour_task=my_task+nproc_x-1
        dest_task=my_task+nproc_x-1

        ! first time to recieve
         call mpi_recv(recvbuf,ny,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do j=1,ny
           wrk(1,j)=recvbuf(j)
         end do


        ! first time to send
         do j=1,ny
           sendbuf(j)=wrk(3,j)
         end do
         call mpi_send(sendbuf,ny,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

        ! second time to recieve
         call mpi_recv(recvbuf,ny,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)

         do j=1,ny
           wrk(2,j)=recvbuf(j)
         end do

      endif !if(n_west.eq.-1)

      endif !if (nproc_x.eq.1) then
*/

	if (nproc_x == 1){
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
	}else{
		if (n_east == -1){
			dest_task = my_task-nproc_x+1;	
			sour_task = my_task-nproc_x+1;

			for (j = 0; j < ny; j++){
				sendbuf[j] = work[j][nx-3];	
			}

			MPI_Send(sendbuf, ny, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (j = 0; j < ny; j++){
				work[j][nx-1] = recvbuf[j];	
			}

			for (j = 0; j < ny; j++){
				sendbuf[j] = work[j][nx-2];	
			}
			
			MPI_Send(sendbuf, ny, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);
		}

		if (n_west == -1){
			sour_task = my_task+nproc_x-1;// ie. n_east == -1 and the same j
			dest_task = my_task+nproc_x-1;

			MPI_Recv(recvbuf, ny, MPI_FLOAT, 
					 sour_task, sour_task, pom_comm, &status);

			for (j = 0; j < ny; j++){
				work[j][0] = recvbuf[j];		
			}

			for (j = 0; j < ny; j++){
				sendbuf[j] = work[j][2];	
			}

			MPI_Send(sendbuf, ny, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny, MPI_FLOAT, 
					 sour_task, sour_task, pom_comm, &status);

			for (j = 0; j < ny; j++){
				work[j][1] = recvbuf[j];	
			}
		}
	}

	/*
      return
      end
	*/
	free(sendbuf);
	free(recvbuf);

	return;

}

/*
      subroutine yperi2d_mpi(wrk,nx,ny)
! doing periodic bc in y
! pass from north to south and also pass from south to north
      implicit none
      include 'mpif.h'
      include 'pom.h'
      integer nx,ny
      real wrk(nx,ny)
      integer i,j,k
      integer ierr,nproc_x,nproc_y
      integer dest_task,sour_task
      integer istatus(mpi_status_size)
      real sendbuf(nx),recvbuf(nx)
*/

void yperi2d_mpi(float work[][i_size],
				 int nx, int ny){
	int i, j, k;
	int dest_task, sour_task;
	int nproc_x, nproc_y;
	float *sendbuf = (float*)malloc(nx*sizeof(float));
	float *recvbuf = (float*)malloc(nx*sizeof(float));
	MPI_Status status;

/*
! determine the number of processors in y
      if(mod(jm_global-2,jm_local-2).eq.0) then
        nproc_y=(jm_global-2)/(jm_local-2)
      else
        nproc_y=(jm_global-2)/(jm_local-2) + 1
      end if
*/
	if ((jm_global-2) % (jm_local-2) == 0){
		nproc_y = (jm_global-2)/(jm_local-2);
	}else{
		nproc_y = (jm_global-2)/(jm_local-2)+1;
	}

	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

/*
      if (nproc_y.eq.1) then
        do i=1,nx
        wrk(i,ny)=wrk(i,3); wrk(i,1)=wrk(i,ny-2); wrk(i,2)=wrk(i,ny-1)
        enddo
      else
!
C  !The most north sudomains    
      if(n_north.eq.-1) then
        dest_task=my_task-nproc_y+1
        sour_task=my_task-nproc_y+1

       ! first time to send
         do i=1,nx
           sendbuf(i)=wrk(i,ny-2)
         end do
         call mpi_send(sendbuf,nx,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)


       !first time to recieve
         call mpi_recv(recvbuf,nx,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do i=1,nx
          wrk(i,ny)=recvbuf(i)
         end do

       ! second time to send
         do i=1,nx
           sendbuf(i)=wrk(i,ny-1)
         end do
         call mpi_send(sendbuf,nx,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

      endif !if(n_north.eq.-1)

C  !The most south sudomains    
      if(n_south.eq.-1) then
        sour_task=my_task+nproc_y-1
        dest_task=my_task+nproc_y-1

        ! first time to recieve
         call mpi_recv(recvbuf,nx,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do i=1,nx
           wrk(i,1)=recvbuf(i)
         end do


        ! first time to send
         do i=1,nx
           sendbuf(i)=wrk(i,3)
         end do
         call mpi_send(sendbuf,nx,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

        ! second time to recieve
         call mpi_recv(recvbuf,nx,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)


         do i=1,nx
           wrk(i,2)=recvbuf(i)
         end do

      endif !if(n_south.eq.-1)

      endif !if (nproc_y.eq.1) then
*/
	if (nproc_y == 1){
		for (i = 0; i < nx; i++){
			work[ny-1][i] = work[2][i];	
			work[0][i] = work[ny-3][i];
			work[1][i] = work[ny-2][i];
		}
	}else{
		if (n_north == -1){
			dest_task = my_task-(nproc_y-1)*nproc_x;	
			sour_task = my_task-(nproc_y-1)*nproc_x;	

			for (i = 0; i < nx; i++){
				sendbuf[i] = work[ny-3][i];	
			}

			MPI_Send(sendbuf, nx, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx, MPI_FLOAT, 
					 sour_task, sour_task, pom_comm, &status);

			for (i = 0; i < nx; i++){
				work[ny-1][i] = recvbuf[i];	
			}

			for (i = 0; i < nx; i++){
				sendbuf[i] = work[ny-2][i];	
			}

			MPI_Send(sendbuf, nx, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);
		}

		if (n_south == -1){
			sour_task = my_task+(nproc_y-1)*nproc_x;
			dest_task = my_task+(nproc_y-1)*nproc_x;

			MPI_Recv(recvbuf, nx, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (i = 0; i < nx; i++){
				work[0][i] = recvbuf[i];	
			}

			for (i = 0; i < nx; i++){
				sendbuf[i] = work[2][i];	
			}

			MPI_Send(sendbuf, nx, MPI_FLOAT,
					dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (i = 0; i < nx; i++){
				work[1][i] = recvbuf[i];	
			}
		}
	}
	free(sendbuf);
	free(recvbuf);

	/*
      return
      end
	*/
	return;
}



/*
!_______________________________________________________________________
      subroutine xperi3d_mpi(wrk,nx,ny,nz)
! doing periodic bc in x
! pass from east to west and also pass from west to east
      implicit none
      include 'mpif.h'
      include 'pom.h'
      integer nx,ny,nz
      real wrk(nx,ny,nz)
      integer i,j,k
      integer ierr,nproc_x,nproc_y
      integer dest_task,sour_task
      integer istatus(mpi_status_size)
      real sendbuf(ny*nz),recvbuf(ny*nz)
*/

void xperi3d_mpi(float work[][j_size][i_size], 
				 int nx, int ny, int nz){

	int i, j, k;
	int nproc_x, nproc_y;
	int dest_task, sour_task;
	MPI_Status status;
	float *sendbuf = (float*)malloc(ny*nz*sizeof(float));
	float *recvbuf = (float*)malloc(ny*nz*sizeof(float));
/*
! determine the number of processors in x
      if(mod(im_global-2,im_local-2).eq.0) then
        nproc_x=(im_global-2)/(im_local-2)
      else
        nproc_x=(im_global-2)/(im_local-2) + 1
      end if

*/
	if ((im_global-2)%(im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}


/*
      if (nproc_x.eq.1) then
        do k=1,nz; do j=1,ny
        wrk(nx,j,k)=wrk(3,j,k);
        wrk(1,j,k)=wrk(nx-2,j,k); wrk(2,j,k)=wrk(nx-1,j,k);
        enddo; enddo
      else
!
C  !The most east sudomains    
      if(n_east.eq.-1) then
        dest_task=my_task-nproc_x+1
        sour_task=my_task-nproc_x+1

        ! first time to send
         do k=1,nz
          do j=1,ny
           i=j+(k-1)*ny
           sendbuf(i)=wrk(nx-2,j,k)
          end do
         end do
         call mpi_send(sendbuf,ny*nz,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

        ! first time to recieve
         call mpi_recv(recvbuf,ny*nz,mpi_real,sour_task,sour_task,
     $                 pom_comm,istatus,ierr)
         do k=1,nz
          do j=1,ny
           i=j+(k-1)*ny
           wrk(nx,j,k)=recvbuf(i)
          end do
         end do

        ! second time to send
         do k=1,nz
          do j=1,ny
           i=j+(k-1)*ny
           sendbuf(i)=wrk(nx-1,j,k)
          end do
         end do
         call mpi_send(sendbuf,ny*nz,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

      endif!if(n_east.eq.-1)

C  !The most west sudomains    
      if(n_west.eq.-1) then
       sour_task=my_task+nproc_x-1
       dest_task=my_task+nproc_x-1

        ! first time to recieve
         call mpi_recv(recvbuf,ny*nz,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do k=1,nz
          do j=1,ny
           i=j+(k-1)*ny
            wrk(1,j,k)=recvbuf(i)
          end do
         end do


        ! first time to send
         do k=1,nz
          do j=1,ny
           i=j+(k-1)*ny
           sendbuf(i)=wrk(3,j,k)
          end do
         end do
         call mpi_send(sendbuf,ny*nz,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

        ! second time to recieve
         call mpi_recv(recvbuf,ny*nz,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do k=1,nz
          do j=1,ny
           i=j+(k-1)*ny
            wrk(2,j,k)=recvbuf(i)
          end do
         end do

      endif!if(n_west.eq.-1)

      endif !if (nproc_x.eq.1) then
*/
	if (nproc_x == 1){
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				work[k][j][nx-1] = work[k][j][2];	
				work[k][j][0] = work[k][j][nx-3];
				work[k][j][1] = work[k][j][nx-2];
			}
		}
	}else{
		if (n_east == -1){
			dest_task = my_task-nproc_x+1;	
			sour_task = my_task-nproc_x+1;

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;
					sendbuf[i] = work[k][j][nx-3];
				}
			}

			MPI_Send(sendbuf, ny*nz, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;
					work[k][j][nx-1] = recvbuf[i];
				}
			}

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;
					sendbuf[i] = work[k][j][nx-2];
				}
			}
			
			MPI_Send(sendbuf, ny*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);
		}

		if (n_west == -1){
			sour_task = my_task+nproc_x-1;
			dest_task = my_task+nproc_x-1;

			MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;	
					work[k][j][0] = recvbuf[i];
				}
			}

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;	
					sendbuf[i] = work[k][j][2];
				}
			}

			MPI_Send(sendbuf, ny*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;	
					work[k][j][1] = recvbuf[i];
				}
			}
		}

	}
	
	free(sendbuf);
	free(recvbuf);

/*
      return
      end
*/
	return;

}
	
/*
!_______________________________________________________________________
      subroutine yperi3d_mpi(wrk,nx,ny,nz)
! doing periodic bc in y
! pass from north to south and also pass from south to north
      implicit none
      include 'mpif.h'
      include 'pom.h'
      integer nx,ny,nz
      real wrk(nx,ny,nz)
      integer i,j,k
      integer ierr,nproc_x,nproc_y
      integer dest_task,sour_task
      integer istatus(mpi_status_size)
      real sendbuf(nx*nz),recvbuf(nx*nz)
*/

void yperi3d_mpi(float work[][j_size][i_size],
				 int nx, int ny, int nz){
	int i, j, k;
	int nproc_x, nproc_y;
	int dest_task, sour_task;
	MPI_Status status;
	float *sendbuf = (float*)malloc(nx*nz*sizeof(float));
	float *recvbuf = (float*)malloc(nx*nz*sizeof(float));

/*
! determine the number of processors in y
      if(mod(jm_global-2,jm_local-2).eq.0) then
        nproc_y=(jm_global-2)/(jm_local-2)
      else
        nproc_y=(jm_global-2)/(jm_local-2) + 1
      end if
*/
	if ((jm_global-2)%(jm_local-2) == 0){
		nproc_y=(jm_global-2)/(jm_local-2);
	}else{
		nproc_y=(jm_global-2)/(jm_local-2)+1;
	}

/*
!lyo:scs1d:
      if (nproc_y.eq.1) then
        do k=1,nz; do i=1,nx
        wrk(i,ny,k)=wrk(i,3,k);
        wrk(i,1,k)=wrk(i,ny-2,k); wrk(i,2,k)=wrk(i,ny-1,k)
        enddo; enddo
      else
!
C  !The most north sudomains    
      if(n_north.eq.-1) then
        dest_task=my_task-nproc_y+1
        sour_task=my_task-nproc_y+1

        ! first time to send
         do k=1,nz
          do i=1,nx
           j=i+(k-1)*nx
           sendbuf(j)=wrk(i,ny-2,k)
          end do
         end do
         call mpi_send(sendbuf,nx*nz,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

        ! first time to recieve
         call mpi_recv(recvbuf,nx*nz,mpi_real,sour_task,sour_task,
     $                 pom_comm,istatus,ierr)
         do k=1,nz
          do i=1,nx
           j=i+(k-1)*nx
           wrk(i,ny,k)=recvbuf(j)
          end do
         end do

        ! second time to send
         do k=1,nz
          do i=1,nx
           j=i+(k-1)*nx
           sendbuf(j)=wrk(i,ny-1,k)
          end do
         end do
         call mpi_send(sendbuf,nx*nz,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

      endif!if(n_north.eq.-1)

C  !The most south sudomains    
      if(n_south.eq.-1) then
       sour_task=my_task+nproc_y-1
       dest_task=my_task+nproc_y-1

        ! first time to recieve
         call mpi_recv(recvbuf,nx*nz,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do k=1,nz
          do i=1,nx
           j=i+(k-1)*nx
            wrk(i,1,k)=recvbuf(j)
          end do
         end do


        ! first time to send
         do k=1,nz
          do i=1,nx
           j=i+(k-1)*nx
           sendbuf(j)=wrk(i,3,k)
          end do
         end do
         call mpi_send(sendbuf,nx*nz,mpi_real,dest_task,my_task,
     $                   pom_comm,ierr)

        ! second time to recieve
         call mpi_recv(recvbuf,nx*nz,mpi_real,sour_task,sour_task,
     $                pom_comm,istatus,ierr)
         do k=1,nz
          do i=1,nx
           j=i+(k-1)*nx
            wrk(i,2,k)=recvbuf(j)
          end do
         end do

      endif!if(n_south.eq.-1)

      endif !if (nproc_y.eq.1) then
*/

	if (nproc_y == 1){
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				work[k][ny-1][i] = work[k][2][i];
				work[k][0][i] = work[k][ny-3][i];
				work[k][1][i] = work[k][ny-2][i];
			}
		}
	}else{
		if (n_north == -1){
			dest_task = my_task-(nproc_y-1)*nproc_x;
			sour_task = my_task-(nproc_y-1)*nproc_x;

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					sendbuf[j] = work[k][ny-3][i];
				}
			}

			MPI_Send(sendbuf, nx*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					work[k][ny-1][i] = recvbuf[j];
				}
			}

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					sendbuf[j] = work[k][ny-2][i];
				}
			}

			MPI_Send(sendbuf, nx*nz, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);
		}

		if (n_south == -1){
			sour_task = my_task+(nproc_y-1)*nproc_x;
			dest_task = my_task+(nproc_y-1)*nproc_x;

			MPI_Recv(recvbuf, nx*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					work[k][0][i] = recvbuf[j];
				}
			}

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					sendbuf[j] = work[k][2][i];
				}
			}

			MPI_Send(sendbuf, nx*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					work[k][1][i] = recvbuf[j];	
				}
			}
		}
	}

	free(sendbuf);
	free(recvbuf);

	/*
      return
      end
	*/
	return;
}


void exchange3d_mpi_tag(float work[][j_size][i_size], 
						int nx, int ny, int nz, int tag){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_mpi, 
				   time_end_exchange3d_mpi;
	timer_now(&time_start_exchange3d_mpi);
#endif
	int i, j, k;
	MPI_Status status;

	
	float *send_east = (float*)malloc(ny*nz*sizeof(float));
	float *recv_west= (float*)malloc(ny*nz*sizeof(float));
	float *send_west = (float*)malloc(ny*nz*sizeof(float));
	float *recv_east= (float*)malloc(ny*nz*sizeof(float));
	float *send_north= (float*)malloc(nx*nz*sizeof(float));
	float *recv_south= (float*)malloc(nx*nz*sizeof(float));
	float *send_south= (float*)malloc(nx*nz*sizeof(float));
	float *recv_north= (float*)malloc(nx*nz*sizeof(float));
	
	//printf("in exchange3d: I come to here0\n");
	if (n_east != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+k*ny;
				send_east[i] = work[k][j][nx-2];
			}
		}
		MPI_Send(send_east, ny*nz, MPI_FLOAT, n_east, tag, pom_comm);
	}

	if (n_west != -1){
		MPI_Recv(recv_west, ny*nz, MPI_FLOAT, n_west, tag, pom_comm, &status);	
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work[k][j][0] = recv_west[i];
			}
		}
	}

// send ghost cell data to the west
	if (n_west != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i =j+k*ny;
				send_west[i] = work[k][j][1];
			}
		}
		MPI_Send(send_west, ny*nz, MPI_FLOAT, n_west, tag, pom_comm);
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		MPI_Recv(recv_east, ny*nz, MPI_FLOAT, n_east, tag, pom_comm, &status);
		for (k = 0; k < nz ; k++){
#pragma simd
#pragma vector
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				work[k][j][nx-1] = recv_east[i];
			}
		}
	}

// send ghost cell data to the north
	if (n_north != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				send_north[j] = work[k][ny-2][i];
			}
		}
		MPI_Send(send_north, nx*nz, MPI_FLOAT, n_north, tag, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx*nz, MPI_FLOAT, n_south, tag, pom_comm, &status);
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				work[k][0][i] = recv_south[j];
			}
		}
	}

// send ghost cell data to the south
	if (n_south != -1){
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				send_south[j] = work[k][1][i];
			}
		}
		MPI_Send(send_south, nx*nz, MPI_FLOAT, n_south, tag, pom_comm);
	}

// recieve ghost cell data from the north
	if (n_north != -1){
		MPI_Recv(recv_north, nx*nz, MPI_FLOAT, n_north, tag, pom_comm, &status);
		for (k = 0; k < nz; k++){
#pragma simd
#pragma vector
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				work[k][ny-1][i] = recv_north[j];
			}
		}
	}

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);

#ifndef TIME_DISABLE
	timer_now(&time_end_exchange3d_mpi);
	exchange3d_mpi_time += time_consumed(&time_start_exchange3d_mpi,
									     &time_end_exchange3d_mpi);
#endif

	return;

}

void exchange2d_mpi_tag(float work[][i_size], int nx, int ny, int tag){
//void exchange2d_mpi_xsz_(float work[][i_size], int *f_nx, int *f_ny){

	/*
	int nx = *f_nx;
	int ny = *f_ny;
	*/

#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_mpi, 
				   time_end_exchange2d_mpi;
	timer_now(&time_start_exchange2d_mpi);
#endif

	float *send_east = (float*)malloc(ny*sizeof(float));
	float *recv_west = (float*)malloc(ny*sizeof(float));
	float *send_west = (float*)malloc(ny*sizeof(float));
	float *recv_east = (float*)malloc(ny*sizeof(float));

	float *send_north = (float*)malloc(nx*sizeof(float));
	float *recv_south = (float*)malloc(nx*sizeof(float));
	float *send_south = (float*)malloc(nx*sizeof(float));
	float *recv_north = (float*)malloc(nx*sizeof(float));

	int i,j;
	MPI_Status status;
//! send ghost cell data to the east
/*
      if(n_east.ne.-1) then
        do j=1,ny
          send_east(j)=work(nx-1,j)
        end do
        call mpi_send(send_east,ny,mpi_real,n_east,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(nx-1, 1), 1, Work_vector, n_east, my_task, 
!     $                pom_comm, ierr)
      end if
*/
	if (n_east != -1){
		for (j = 0; j < ny; j++){
			send_east[j] = work[j][nx-2];	
		}
		MPI_Send(send_east, ny, MPI_FLOAT, n_east, tag, pom_comm);
	}

//! recieve ghost cell data from the west
/*
      if(n_west.ne.-1) then
        call mpi_recv(recv_west,ny,mpi_real,n_west,n_west,
     $                pom_comm,istatus,ierr)
        do j=1,ny
          work(1,j)=recv_west(j)
        end do
!        call MPI_RECV(work(1, 1), 1, Work_vector, n_west, n_west, 
!     $                pom_comm, istatus, ierr)
      end if
*/
	if (n_west != -1){
		MPI_Recv(recv_west, ny, MPI_FLOAT, n_west, tag, pom_comm, &status);	
		for (j = 0; j < ny; j++){
			work[j][0] = recv_west[j];
		}
	}

//! send ghost cell data to the west
/*
      if(n_west.ne.-1) then
        do j=1,ny
          send_west(j)=work(2,j)
        end do
        call mpi_send(send_west,ny,mpi_real,n_west,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(2,1), 1, Work_vector, n_west, my_task,
!     $                pom_comm,ierr)
      end if
*/

	if (n_west != -1){
		for (j = 0; j < ny; j++){
			send_west[j] = work[j][1];
		}
		MPI_Send(send_west, ny, MPI_FLOAT, n_west, tag, pom_comm);
	}

//! recieve ghost cell data from the east
/*
      if(n_east.ne.-1) then
        call mpi_recv(recv_east,ny,mpi_real,n_east,n_east,
     $                pom_comm,istatus,ierr)
        do j=1,ny
          work(nx,j)=recv_east(j)
        end do
!        call MPI_RECV(work(nx, 1), 1, Work_vector, n_east, n_east,
!     $                pom_comm,istatus,ierr)
      end if
*/
	if (n_east != -1){
		MPI_Recv(recv_east, ny, MPI_FLOAT, n_east, tag, pom_comm, &status);	
		for (j = 0; j < ny; j++){
			work[j][nx-1] = recv_east[j];	
		}
	}

//! send ghost cell data to the north
/*
      if(n_north.ne.-1) then
        do i=1,nx
          send_north(i)=work(i,ny-1)
        end do
        call mpi_send(send_north,nx,mpi_real,n_north,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(1, ny-1),nx,mpi_real,n_north,my_task,
!     $                pom_comm,ierr)
      end if
*/
	if (n_north != -1){
		for (i = 0; i < nx; i++){
			send_north[i] = work[ny-2][i];	
		}
		MPI_Send(send_north, nx, MPI_FLOAT, n_north, tag, pom_comm);
	}

//! recieve ghost cell data from the south
/*
      if(n_south.ne.-1) then
        call mpi_recv(recv_south,nx,mpi_real,n_south,n_south,
     $                pom_comm,istatus,ierr)
        do i=1,nx
          work(i,1)=recv_south(i)
        end do
!        call MPI_RECV(work(1, 1),nx,mpi_real,n_south,n_south,
!     $                pom_comm,istatus,ierr)
      

      end if
*/
	if (n_south != -1){
		MPI_Recv(recv_south, nx, MPI_FLOAT, n_south, tag, pom_comm, &status);	
		for (i = 0; i < nx; i++){
			work[0][i] = recv_south[i];	
		}
	}

//! send ghost cell data to the south
/*
      if(n_south.ne.-1) then
        do i=1,nx
          send_south(i)=work(i,2)
        end do
        call mpi_send(send_south,nx,mpi_real,n_south,my_task,
     $                pom_comm,ierr)
!        call MPI_SEND(work(1, 2),nx,mpi_real,n_south,my_task,
!     $                pom_comm,ierr)
      end if
*/
	if (n_south != -1){
		for (i = 0; i < nx; i++){
			send_south[i] = work[1][i];	
		}
		MPI_Send(send_south, nx, MPI_FLOAT, n_south, tag, pom_comm);
	}

//! recieve ghost cell data from the north
/*
      if(n_north.ne.-1) then
        call mpi_recv(recv_north,nx,mpi_real,n_north,n_north,
     $                pom_comm,istatus,ierr)
        do i=1,nx
          work(i,ny)=recv_north(i)
        end do
!        call MPI_RECV(work(1, ny),nx,mpi_real,n_north,n_north,
!     $                pom_comm,istatus,ierr)
      end if
*/
	if (n_north != -1){
		MPI_Recv(recv_north, nx, MPI_FLOAT, n_north, tag, pom_comm, &status);
		for (i = 0; i < nx; i++){
			work[ny-1][i] = recv_north[i];	
		}
	}

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);
#ifndef TIME_DISABLE
	timer_now(&time_end_exchange2d_mpi);
	exchange2d_mpi_time += time_consumed(&time_start_exchange2d_mpi,
									     &time_end_exchange2d_mpi);
#endif

	return;
}
