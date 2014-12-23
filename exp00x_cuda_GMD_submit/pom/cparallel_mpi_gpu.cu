#include<stdio.h>
#include<mpi.h>

#include"cparallel_mpi_gpu.h"
#include"cu_data.h"

extern "C"{
	#include"data.h"
	#include"timer_all.h"
	#include"cparallel_mpi.h"
}

//void exchange3d_mpi(float work[][j_size][i_size], int nx, int ny, int nz){
//void exchange3d_mpi_gpu(float work[][j_size][i_size], 
//						  float *d_work, int nx, int ny, int nz){
void exchange3d_mpi_gpu(float *d_work, int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d,
				   time_end_exchange3d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d);
#endif

	if (n_proc != 1){

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
	
#ifndef CUDA_SLICE_MPI
	/*
	float (*h_work)[j_size][i_size] = 
		(float(*)[j_size][i_size])malloc(nx*ny*nz*sizeof(float));
	*/
	float h_work[k_size][j_size][i_size];

	checkCudaErrors(cudaMemcpy(h_work, d_work, nx*ny*nz*sizeof(float),
							   cudaMemcpyDeviceToHost));
#endif

#ifdef PURE_GPU_MPI
	exchange3d_mpi(h_work, nx, ny, nz);

#else

	//printf("in exchange3d: I come to here0\n");
	if (n_east != -1){
		
#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(send_east, sizeof(float), d_work+(nx-2), nx*sizeof(float), sizeof(float), ny*nz, cudaMemcpyDeviceToHost));
#else
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;
				send_east[i] = h_work[k][j][nx-2];
			}
		}
#endif
		
		MPI_Send(send_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm);
	}

	if (n_west != -1){
		MPI_Recv(recv_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &status);	

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float), recv_west, sizeof(float), sizeof(float), ny*nz, cudaMemcpyHostToDevice));
#else
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				h_work[k][j][0] = recv_west[i];
			}
		}
#endif
		
	}

// send ghost cell data to the west
	if (n_west != -1){
		
#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(send_west, sizeof(float), d_work+1, nx*sizeof(float), sizeof(float), ny*nz, cudaMemcpyDeviceToHost));

#else

		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				i =j+k*ny;
				send_west[i] = h_work[k][j][1];
			}
		}
#endif
		
		MPI_Send(send_west, ny*nz, MPI_FLOAT, n_west, my_task, pom_comm);
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		MPI_Recv(recv_east, ny*nz, MPI_FLOAT, n_east, n_east, pom_comm, &status);

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(d_work+(nx-1), nx*sizeof(float), recv_east, sizeof(float), sizeof(float), ny*nz, cudaMemcpyHostToDevice));
		
#else
		for (k = 0; k < nz ; k++){
			for (j = 0; j < ny; j++){
				i = j+k*ny;	
				h_work[k][j][nx-1] = recv_east[i];
			}
		}
#endif

	}

// send ghost cell data to the north
	if (n_north != -1){

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(send_north, nx*sizeof(float), d_work+(ny-2)*nx, ny*nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyDeviceToHost));
#else
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				send_north[j] = h_work[k][ny-2][i];
			}
		}
#endif
		MPI_Send(send_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Recv(recv_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &status);

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(d_work, ny*nx*sizeof(float), recv_south, nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyHostToDevice));
#else

		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				h_work[k][0][i] = recv_south[j];
			}
		}
#endif

	}

// send ghost cell data to the south
	if (n_south != -1){

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(send_south, nx*sizeof(float), d_work+nx, ny*nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyDeviceToHost));

#else
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;
				send_south[j] = h_work[k][1][i];
			}
		}
#endif

		MPI_Send(send_south, nx*nz, MPI_FLOAT, n_south, my_task, pom_comm);
	}

// recieve ghost cell data from the north
	if (n_north != -1){
		MPI_Recv(recv_north, nx*nz, MPI_FLOAT, n_north, n_north, pom_comm, &status);

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(d_work+(ny-1)*nx, ny*nx*sizeof(float), recv_north, nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyHostToDevice));

#else
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				j = i+k*nx;	
				h_work[k][ny-1][i] = recv_north[j];
			}
		}
#endif
	}

#endif

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);

#ifndef CUDA_SLICE_MPI
	checkCudaErrors(cudaMemcpy(d_work, h_work, nx*ny*nz*sizeof(float),
							   cudaMemcpyHostToDevice));
	//free(h_work);
#endif
	}


#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d);
	exchange3d_mpi_time += time_consumed(&time_start_exchange3d,
							             &time_end_exchange3d);
#endif
	return;

}


void exchange2d_mpi_gpu(float *d_work, int nx, int ny){
//void exchange2d_mpi_gpu(float *d_work, int nx, int ny){
//void exchange2d_mpi_xsz_(float work[][i_size], int *f_nx, int *f_ny){

	/*
	int nx = *f_nx;
	int ny = *f_ny;
	*/

#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d,
				   time_end_exchange2d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d);
#endif
	if (n_proc != 1){

	int i, j;
	MPI_Status status;

	float *send_east = (float*)malloc(ny*sizeof(float));
	float *recv_west = (float*)malloc(ny*sizeof(float));
	float *send_west = (float*)malloc(ny*sizeof(float));
	float *recv_east = (float*)malloc(ny*sizeof(float));

	float *send_north = (float*)malloc(nx*sizeof(float));
	float *recv_south = (float*)malloc(nx*sizeof(float));
	float *send_south = (float*)malloc(nx*sizeof(float));
	float *recv_north = (float*)malloc(nx*sizeof(float));

#ifndef CUDA_SLICE_MPI
	/*
	float (*h_work)[i_size] = 
			(float(*)[i_size])malloc(nx*ny*sizeof(float));
	*/
	float h_work[j_size][i_size];

	checkCudaErrors(cudaMemcpy(h_work, d_work, nx*ny*sizeof(float),
							   cudaMemcpyDeviceToHost));
#endif

#ifdef PURE_GPU_MPI
	exchange2d_mpi(h_work, nx, ny);

#else
//! send ghost cell data to the east

	if (n_east != -1){

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(send_east, sizeof(float), d_work+(nx-2), nx*sizeof(float), sizeof(float), ny, cudaMemcpyDeviceToHost));
#else
		for (j = 0; j < ny; j++){
			send_east[j] = h_work[j][nx-2];	
		}

#endif
		MPI_Send(send_east, ny, MPI_FLOAT, n_east, my_task, pom_comm);
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		MPI_Recv(recv_west, ny, MPI_FLOAT, n_west, n_west, pom_comm, &status);	

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float), recv_west, sizeof(float), sizeof(float), ny, cudaMemcpyHostToDevice));
#else

		for (j = 0; j < ny; j++){
			h_work[j][0] = recv_west[j];
		}
#endif

	}

//! send ghost cell data to the west

	if (n_west != -1){
#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(send_west, sizeof(float), d_work+1, nx*sizeof(float), sizeof(float), ny, cudaMemcpyDeviceToHost));
#else
		for (j = 0; j < ny; j++){
			send_west[j] = h_work[j][1];
		}
#endif
		MPI_Send(send_west, ny, MPI_FLOAT, n_west, my_task, pom_comm);
	}

//! recieve ghost cell data from the east

	if (n_east != -1){
		MPI_Recv(recv_east, ny, MPI_FLOAT, n_east, n_east, pom_comm, &status);	

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy2D(d_work+(nx-1), nx*sizeof(float), recv_east, sizeof(float), sizeof(float), ny, cudaMemcpyHostToDevice));
#else
		for (j = 0; j < ny; j++){
			h_work[j][nx-1] = recv_east[j];	
		}
#endif
	}

//! send ghost cell data to the north
	if (n_north != -1){
#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy(send_north, d_work+(ny-2)*nx, nx*sizeof(float), cudaMemcpyDeviceToHost));
#else
		for (i = 0; i < nx; i++){
			send_north[i] = h_work[ny-2][i];	
		}
#endif
		MPI_Send(send_north, nx, MPI_FLOAT, n_north, my_task, pom_comm);
	}

//! recieve ghost cell data from the south
	if (n_south != -1){

		MPI_Recv(recv_south, nx, MPI_FLOAT, n_south, n_south, pom_comm, &status);	

#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy(d_work, recv_south, nx*sizeof(float), cudaMemcpyHostToDevice));
#else
		for (i = 0; i < nx; i++){
			h_work[0][i] = recv_south[i];	
		}
#endif
	}

//! send ghost cell data to the south
	if (n_south != -1){
#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy(send_north, d_work+nx, nx*sizeof(float), cudaMemcpyDeviceToHost));
#else
		for (i = 0; i < nx; i++){
			send_south[i] = h_work[1][i];	
		}
#endif
		MPI_Send(send_south, nx, MPI_FLOAT, n_south, my_task, pom_comm);
	}

//! recieve ghost cell data from the north
	if (n_north != -1){
		MPI_Recv(recv_north, nx, MPI_FLOAT, n_north, n_north, pom_comm, &status);
#ifdef CUDA_SLICE_MPI
		checkCudaErrors(cudaMemcpy(d_work+(ny-1)*nx, recv_north, nx*sizeof(float), cudaMemcpyHostToDevice));
#else
		for (i = 0; i < nx; i++){
			h_work[ny-1][i] = recv_north[i];	
		}
#endif
	}

#endif //pure_gpu_mpi

	free(send_east);
	free(recv_west);
	free(send_west);
	free(recv_east);
	free(send_north);
	free(recv_south);
	free(send_south);
	free(recv_north);

#ifndef CUDA_SLICE_MPI
	checkCudaErrors(cudaMemcpy(d_work, h_work, nx*ny*sizeof(float),
							   cudaMemcpyHostToDevice));
	//free(h_work);
#endif
	}

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d);
	exchange2d_mpi_time += time_consumed(&time_start_exchange2d,
							             &time_end_exchange2d);
#endif

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
void xperi2d_mpi_gpu(float *d_work, int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d,
				   time_end_xperi2d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d);
#endif

	int j;
	int nproc_x;
	int dest_task, sour_task;
	float (*h_work)[i_size] = (float(*)[i_size])malloc(nx*ny*sizeof(float));
	float *sendbuf = (float*)malloc(ny*sizeof(float));
	float *recvbuf = (float*)malloc(ny*sizeof(float));

	MPI_Status status;

	float test = 1.f;
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
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

	}else{
		if (n_east == -1){
			
			checkCudaErrors(cudaMemcpy(h_work, d_work, ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			dest_task = my_task-nproc_x+1;	
			sour_task = my_task-nproc_x+1;

			for (j = 0; j < ny; j++){
				sendbuf[j] = h_work[j][nx-3];	
			}

			//printf("rank:%d east_xperi2d_mpi_1, dest_task = %d\n", my_task,
			//		dest_task);
			MPI_Send(sendbuf, ny, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);
			//MPI_Send(&test, 1, MPI_FLOAT, 
			//		 0, 0, MPI_COMM_WORLD);

			MPI_Recv(recvbuf, ny, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (j = 0; j < ny; j++){
				h_work[j][nx-1] = recvbuf[j];	
			}

			for (j = 0; j < ny; j++){
				sendbuf[j] = h_work[j][nx-2];	
			}
			
			MPI_Send(sendbuf, ny, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			checkCudaErrors(cudaMemcpy(d_work, h_work, ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}

		if (n_west == -1){

			checkCudaErrors(cudaMemcpy(h_work, d_work, ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			sour_task = my_task+nproc_x-1;// ie. n_east == -1 and the same j
			dest_task = my_task+nproc_x-1;

			//printf("rank:%d west_xperi2d_mpi_1, sour_task = %d\n", my_task, sour_task);
			MPI_Recv(recvbuf, ny, MPI_FLOAT, 
					 sour_task, sour_task, pom_comm, &status);
			//MPI_Recv(&test, 1, MPI_FLOAT, 
			//		 1, 0, MPI_COMM_WORLD, &status);

			for (j = 0; j < ny; j++){
				h_work[j][0] = recvbuf[j];		
			}

			for (j = 0; j < ny; j++){
				sendbuf[j] = h_work[j][2];	
			}

			MPI_Send(sendbuf, ny, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny, MPI_FLOAT, 
					 sour_task, sour_task, pom_comm, &status);

			for (j = 0; j < ny; j++){
				h_work[j][1] = recvbuf[j];	
			}

			checkCudaErrors(cudaMemcpy(d_work, h_work, ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}
	}

	/*
      return
      end
	*/
	free(h_work);
	free(sendbuf);
	free(recvbuf);

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d);
	xperi2d_mpi_time += time_consumed(&time_start_xperi2d,
							          &time_end_xperi2d);
#endif

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

void yperi2d_mpi_gpu(float *d_work, int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_yperi2d,
				   time_end_yperi2d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_yperi2d);
#endif

	int i;
	int dest_task, sour_task;
	int nproc_x, nproc_y;

	float (*h_work)[i_size] = (float(*)[i_size])malloc(nx*ny*sizeof(float));
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
		/*
		for (i = 0; i < nx; i++){
			work[ny-1][i] = work[2][i];	
			work[0][i] = work[ny-3][i];
			work[1][i] = work[ny-2][i];
		}
		*/

		checkCudaErrors(cudaMemcpy(d_work+(ny-1)*im, d_work+2*im,
								   nx*sizeof(float),
								   cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy(d_work, d_work+(ny-3)*im,
								   nx*sizeof(float),
								   cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy(d_work+im, d_work+(ny-2)*im,
								   nx*sizeof(float),
								   cudaMemcpyDeviceToDevice));
	}else{
		if (n_north == -1){

			checkCudaErrors(cudaMemcpy(h_work, d_work, ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			dest_task = my_task-(nproc_y-1)*nproc_x;	
			sour_task = my_task-(nproc_y-1)*nproc_x;	

			for (i = 0; i < nx; i++){
				sendbuf[i] = h_work[ny-3][i];	
			}

			MPI_Send(sendbuf, nx, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx, MPI_FLOAT, 
					 sour_task, sour_task, pom_comm, &status);

			for (i = 0; i < nx; i++){
				h_work[ny-1][i] = recvbuf[i];	
			}

			for (i = 0; i < nx; i++){
				sendbuf[i] = h_work[ny-2][i];	
			}

			MPI_Send(sendbuf, nx, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			checkCudaErrors(cudaMemcpy(d_work, h_work, ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}

		if (n_south == -1){

			checkCudaErrors(cudaMemcpy(h_work, d_work, ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			sour_task = my_task+(nproc_y-1)*nproc_x;
			dest_task = my_task+(nproc_y-1)*nproc_x;

			MPI_Recv(recvbuf, nx, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (i = 0; i < nx; i++){
				h_work[0][i] = recvbuf[i];	
			}

			for (i = 0; i < nx; i++){
				sendbuf[i] = h_work[2][i];	
			}

			MPI_Send(sendbuf, nx, MPI_FLOAT,
					dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (i = 0; i < nx; i++){
				h_work[1][i] = recvbuf[i];	
			}

			checkCudaErrors(cudaMemcpy(d_work, h_work, ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}
	}

	free(h_work);
	free(sendbuf);
	free(recvbuf);

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_yperi2d);
	yperi2d_mpi_time += time_consumed(&time_start_yperi2d,
							          &time_end_yperi2d);
#endif

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

void xperi3d_mpi_gpu(float *d_work,
				 int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi3d,
				   time_end_xperi3d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi3d);
#endif

	int i, j, k;
	int nproc_x;
	int dest_task, sour_task;
	MPI_Status status;

	float (*h_work)[j_size][i_size] 
		= (float(*)[j_size][i_size])malloc(nx*ny*nz*sizeof(float));
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
		/*
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				work[k][j][nx-1] = work[k][j][2];	
				work[k][j][0] = work[k][j][nx-3];
				work[k][j][1] = work[k][j][nx-2];
			}
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

	}else{
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy(h_work, d_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			dest_task = my_task-nproc_x+1;	
			sour_task = my_task-nproc_x+1;

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;
					sendbuf[i] = h_work[k][j][nx-3];
				}
			}

			MPI_Send(sendbuf, ny*nz, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;
					h_work[k][j][nx-1] = recvbuf[i];
				}
			}

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;
					sendbuf[i] = h_work[k][j][nx-2];
				}
			}
			
			MPI_Send(sendbuf, ny*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			checkCudaErrors(cudaMemcpy(d_work, h_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));

		}

		if (n_west == -1){

			checkCudaErrors(cudaMemcpy(h_work, d_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			sour_task = my_task+nproc_x-1;
			dest_task = my_task+nproc_x-1;

			MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;	
					h_work[k][j][0] = recvbuf[i];
				}
			}

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;	
					sendbuf[i] = h_work[k][j][2];
				}
			}

			MPI_Send(sendbuf, ny*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (j = 0; j < ny; j++){
					i = j+k*ny;	
					h_work[k][j][1] = recvbuf[i];
				}
			}

			checkCudaErrors(cudaMemcpy(d_work, h_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}

	}
	
	free(h_work);
	free(sendbuf);
	free(recvbuf);

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi3d);
	xperi3d_mpi_time += time_consumed(&time_start_xperi3d,
							          &time_end_xperi3d);
#endif

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

void yperi3d_mpi_gpu(float *d_work,
				 int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_yperi3d,
				   time_end_yperi3d;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_yperi3d);
#endif

	int i, j, k;
	int nproc_x, nproc_y;
	int dest_task, sour_task;
	MPI_Status status;


	float (*h_work)[j_size][i_size] 
		= (float(*)[j_size][i_size])malloc(nx*ny*nz*sizeof(float));

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

	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
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
		/*
		for (k = 0; k < nz; k++){
			for (i = 0; i < nx; i++){
				work[k][ny-1][i] = work[k][2][i];
				work[k][0][i] = work[k][ny-3][i];
				work[k][1][i] = work[k][ny-2][i];
			}
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+(ny-1)*im, ny*nx*sizeof(float),
								     d_work+2*im, ny*nx*sizeof(float),
									 nx*sizeof(float), nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, ny*nx*sizeof(float),
								     d_work+(ny-3)*im, ny*nx*sizeof(float),
									 nx*sizeof(float), nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1*im, ny*nx*sizeof(float),
								     d_work+(ny-2)*im, ny*nx*sizeof(float),
									 nx*sizeof(float), nz,
									 cudaMemcpyDeviceToDevice));
	}else{
		if (n_north == -1){
			checkCudaErrors(cudaMemcpy(h_work, d_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			dest_task = my_task-(nproc_y-1)*nproc_x;
			sour_task = my_task-(nproc_y-1)*nproc_x;

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					sendbuf[j] = h_work[k][ny-3][i];
				}
			}

			MPI_Send(sendbuf, nx*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					h_work[k][ny-1][i] = recvbuf[j];
				}
			}

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					sendbuf[j] = h_work[k][ny-2][i];
				}
			}

			MPI_Send(sendbuf, nx*nz, MPI_FLOAT, 
					 dest_task, my_task, pom_comm);

			checkCudaErrors(cudaMemcpy(d_work, h_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}

		if (n_south == -1){
			checkCudaErrors(cudaMemcpy(h_work, d_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyDeviceToHost));

			sour_task = my_task+(nproc_y-1)*nproc_x;
			dest_task = my_task+(nproc_y-1)*nproc_x;

			MPI_Recv(recvbuf, nx*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					h_work[k][0][i] = recvbuf[j];
				}
			}

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					j = i+k*nx;	
					sendbuf[j] = h_work[k][2][i];
				}
			}

			MPI_Send(sendbuf, nx*nz, MPI_FLOAT,
					 dest_task, my_task, pom_comm);

			MPI_Recv(recvbuf, nx*nz, MPI_FLOAT,
					 sour_task, sour_task, pom_comm, &status);

			for (k = 0; k < nz; k++){
				for (i = 0; i < nx; i++){
					h_work[k][1][i] = recvbuf[j];	
				}
			}

			checkCudaErrors(cudaMemcpy(d_work, h_work, 
									   nz*ny*nx*sizeof(float),
								       cudaMemcpyHostToDevice));
		}
	}

	free(h_work);
	free(sendbuf);
	free(recvbuf);

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_yperi3d);
	yperi3d_mpi_time += time_consumed(&time_start_yperi3d,
							          &time_end_yperi3d);
#endif

	/*
      return
      end
	*/
	return;
}


void exchange2d_cuda_aware_mpi(float *d_work, int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_aware,
				   time_end_exchange2d_cuda_aware;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_aware);
#endif

//! send ghost cell data to the east

	float *d_send_to_east = d_1d_ny_tmp0;
	float *d_recv_from_west = d_1d_ny_tmp1;

	float *d_send_to_west = d_1d_ny_tmp2;
	float *d_recv_from_east = d_1d_ny_tmp3;

	MPI_Request request[2];
	MPI_Status status[2];

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2D(d_send_to_east, sizeof(float), 
									 d_work+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDeviceToDevice));
		MPI_Isend(d_send_to_east, ny, MPI_FLOAT, n_east, my_task, 
				  pom_comm, &request[0]);
		MPI_Irecv(d_recv_from_east, ny, MPI_FLOAT, n_east, n_east, 
				  pom_comm, &request[1]);	
		MPI_Waitall(2, request, status);
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2D(d_send_to_west, sizeof(float), 
									 d_work+1, nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDeviceToDevice));
		MPI_Irecv(d_recv_from_west, ny, MPI_FLOAT, n_west, n_west, 
				  pom_comm, &request[0]);	
		MPI_Isend(d_send_to_west, ny, MPI_FLOAT, n_west, my_task, 
				  pom_comm, &request[1]);
		MPI_Waitall(2, request, status);
	}


//! send ghost cell data to the west

	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float), d_recv_from_west, sizeof(float), sizeof(float), ny, cudaMemcpyDeviceToDevice));

	}

//! recieve ghost cell data from the east

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float), d_recv_from_east, sizeof(float), sizeof(float), ny, cudaMemcpyDeviceToDevice));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		MPI_Isend(d_work+(ny-2)*nx, nx, MPI_FLOAT, n_north, my_task, pom_comm, &request[0]);
		MPI_Irecv(d_work+(ny-1)*nx, nx, MPI_FLOAT, n_north, n_north, pom_comm, &request[1]);
		MPI_Waitall(2, request, status);
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		MPI_Irecv(d_work, nx, MPI_FLOAT, n_south, n_south, pom_comm, &request[0]);	
		MPI_Isend(d_work+nx, nx, MPI_FLOAT, n_south, my_task, pom_comm, &request[1]);
		MPI_Waitall(2, request, status);
	}


//! send ghost cell data to the south
//	if (n_south != -1){
//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy(send_north, d_work+nx, nx*sizeof(float), cudaMemcpyDeviceToHost));
//#else
//		for (i = 0; i < nx; i++){
//			send_south[i] = h_work[1][i];	
//		}
//#endif
//		MPI_Send(send_south, nx, MPI_FLOAT, n_south, my_task, pom_comm);
//	}

//! recieve ghost cell data from the north
//	if (n_north != -1){
//		MPI_Recv(recv_north, nx, MPI_FLOAT, n_north, n_north, pom_comm, &status);
//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy(d_work+(ny-1)*nx, recv_north, nx*sizeof(float), cudaMemcpyHostToDevice));
//#else
//		for (i = 0; i < nx; i++){
//			h_work[ny-1][i] = recv_north[i];	
//		}
//#endif
//	}

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_aware);
	exchange2d_cuda_aware_mpi_time += time_consumed(&time_start_exchange2d_cuda_aware,
												    &time_end_exchange2d_cuda_aware);
#endif

	return;
}

void exchange3d_cuda_aware_mpi(float *d_work, int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_cuda_aware,
				   time_end_exchange3d_cuda_aware;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_cuda_aware);
#endif

	int i, j, k;
	MPI_Request request[2];
	MPI_Status status[2];
	
	float *send_to_east = d_2d_ny_nz_tmp0;
	float *recv_from_east = d_2d_ny_nz_tmp1;
	float *send_to_west = d_2d_ny_nz_tmp2;
	float *recv_from_west = d_2d_ny_nz_tmp3;

	float *send_to_north = d_2d_nx_nz_tmp0;
	float *recv_from_north = d_2d_nx_nz_tmp1; 
	float *send_to_south = d_2d_nx_nz_tmp3; 
	float *recv_from_south = d_2d_nx_nz_tmp2; 
	

	if (n_east != -1){
		
		checkCudaErrors(cudaMemcpy2D(send_to_east, sizeof(float), 
									 d_work+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDeviceToDevice));

		MPI_Isend(send_to_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm, &request[0]);
		MPI_Irecv(recv_from_east, ny*nz, MPI_FLOAT, n_east, n_east, pom_comm, &request[1]);
		MPI_Waitall(2, request, status);

//		for (k = 0; k < nz; k++){
//			for (j = 0; j < ny; j++){
//				i = j+k*ny;
//				send_east[i] = h_work[k][j][nx-2];
//			}
//		}
//
//		MPI_Send(send_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm);
	}

	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2D(send_to_west, sizeof(float), 
									 d_work+1, nx*sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDeviceToDevice));

		MPI_Irecv(recv_from_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &request[0]);	
		MPI_Isend(send_to_west, ny*nz, MPI_FLOAT, n_west, my_task, pom_comm, &request[1]);
		MPI_Waitall(2, request, status);


//		MPI_Recv(recv_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &status);	
//		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float), recv_west, sizeof(float), sizeof(float), ny*nz, cudaMemcpyHostToDevice));
//		for (k = 0; k < nz; k++){
//			for (j = 0; j < ny; j++){
//				i = j+k*ny;	
//				h_work[k][j][0] = recv_west[i];
//			}
//		}
		
	}


// send ghost cell data to the west
	if (n_west != -1){
		
		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float), 
									 recv_from_west, sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDeviceToDevice));
//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy2D(send_west, sizeof(float), d_work+1, nx*sizeof(float), sizeof(float), ny*nz, cudaMemcpyDeviceToHost));
//
//#else
//
//		for (k = 0; k < nz; k++){
//			for (j = 0; j < ny; j++){
//				i =j+k*ny;
//				send_west[i] = h_work[k][j][1];
//			}
//		}
//#endif
//		
//		MPI_Send(send_west, ny*nz, MPI_FLOAT, n_west, my_task, pom_comm);
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2D(d_work+(nx-1), nx*sizeof(float), 
									 recv_from_east, sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDeviceToDevice));

//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy2D(d_work+(nx-1), nx*sizeof(float), recv_east, sizeof(float), sizeof(float), ny*nz, cudaMemcpyHostToDevice));
//		
//#else
//		for (k = 0; k < nz ; k++){
//			for (j = 0; j < ny; j++){
//				i = j+k*ny;	
//				h_work[k][j][nx-1] = recv_east[i];
//			}
//		}
//#endif

	}

// send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpy2D(send_to_north, nx*sizeof(float), 
									 d_work+(ny-2)*nx, ny*nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDeviceToDevice));

		MPI_Isend(send_to_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm, &request[0]);
		MPI_Irecv(recv_from_north, nx*nz, MPI_FLOAT, n_north, n_north, pom_comm, &request[1]);
		MPI_Waitall(2, request, status);

//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy2D(send_north, nx*sizeof(float), d_work+(ny-2)*nx, ny*nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyDeviceToHost));
//#else
//		for (k = 0; k < nz; k++){
//			for (i = 0; i < nx; i++){
//				j = i+k*nx;	
//				send_north[j] = h_work[k][ny-2][i];
//			}
//		}
//#endif
//		MPI_Send(send_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpy2D(send_to_south, nx*sizeof(float), 
									 d_work+nx, ny*nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDeviceToDevice));

		MPI_Irecv(recv_from_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &request[0]);
		MPI_Isend(send_to_south, nx*nz, MPI_FLOAT, n_south, my_task, pom_comm, &request[1]);
		MPI_Waitall(2, request, status);

//		MPI_Recv(recv_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &status);
//
//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy2D(d_work, ny*nx*sizeof(float), recv_south, nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyHostToDevice));
//#else
//
//		for (k = 0; k < nz; k++){
//			for (i = 0; i < nx; i++){
//				j = i+k*nx;
//				h_work[k][0][i] = recv_south[j];
//			}
//		}
//#endif

	}


// send ghost cell data to the south
	if (n_south != -1){

		checkCudaErrors(cudaMemcpy2D(d_work, ny*nx*sizeof(float), 
									 recv_from_south, nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDeviceToDevice));

//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy2D(send_south, nx*sizeof(float), d_work+nx, ny*nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyDeviceToHost));
//
//#else
//		for (k = 0; k < nz; k++){
//			for (i = 0; i < nx; i++){
//				j = i+k*nx;
//				send_south[j] = h_work[k][1][i];
//			}
//		}
//#endif
//
//		MPI_Send(send_south, nx*nz, MPI_FLOAT, n_south, my_task, pom_comm);
	}

// recieve ghost cell data from the north
	if (n_north != -1){

		checkCudaErrors(cudaMemcpy2D(d_work+(ny-1)*nx, ny*nx*sizeof(float), 
									 recv_from_north, nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDeviceToDevice));

//		MPI_Recv(recv_north, nx*nz, MPI_FLOAT, n_north, n_north, pom_comm, &status);
//#ifdef CUDA_SLICE_MPI
//		checkCudaErrors(cudaMemcpy2D(d_work+(ny-1)*nx, ny*nx*sizeof(float), recv_north, nx*sizeof(float), nx*sizeof(float), nz, cudaMemcpyHostToDevice));
//
//#else
//		for (k = 0; k < nz; k++){
//			for (i = 0; i < nx; i++){
//				j = i+k*nx;	
//				h_work[k][ny-1][i] = recv_north[j];
//			}
//		}
//#endif
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_cuda_aware);
	exchange3d_cuda_aware_mpi_time += time_consumed(&time_start_exchange3d_cuda_aware,
							             &time_end_exchange3d_cuda_aware);
#endif
	return;

}


void xperi3d_cuda_aware_mpi(float *d_work,
						    int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi3d_cuda_aware,
				   time_end_xperi3d_cuda_aware;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi3d_cuda_aware);
#endif

	int nproc_x;
	int dest_task, sour_task;

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


	if (nproc_x == 1){
		/*
		for (k = 0; k < nz; k++){
			for (j = 0; j < ny; j++){
				work[k][j][nx-1] = work[k][j][2];	
				work[k][j][0] = work[k][j][nx-3];
				work[k][j][1] = work[k][j][nx-2];
			}
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

	}else{
		MPI_Request request[3];
		MPI_Status status[3];
		if (n_east == -1){
			float *east_send_nx_2 = d_2d_ny_nz_tmp0;
			float *east_send_nx_3 = d_2d_ny_nz_tmp1;
			float *east_recv_nx_1 = d_2d_ny_nz_tmp2;

			checkCudaErrors(cudaMemcpy2D(east_send_nx_3, sizeof(float),
									     d_work+nx-3, nx*sizeof(float),
										 sizeof(float), ny*nz,
										 cudaMemcpyDeviceToDevice));

			checkCudaErrors(cudaMemcpy2D(east_send_nx_2, sizeof(float),
									     d_work+nx-2, nx*sizeof(float),
										 sizeof(float), ny*nz,
										 cudaMemcpyDeviceToDevice));
			dest_task = my_task-nproc_x+1;	
			sour_task = my_task-nproc_x+1;

			MPI_Isend(east_send_nx_3, ny*nz, MPI_FLOAT, 
					  dest_task, 0, pom_comm, &request[0]);
			MPI_Isend(east_send_nx_2, ny*nz, MPI_FLOAT, 
					  dest_task, 1, pom_comm, &request[1]);
			MPI_Irecv(east_recv_nx_1, ny*nz, MPI_FLOAT,
					  sour_task, 2, pom_comm, &request[2]);

			MPI_Waitall(3, request, status);

			checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
									     east_recv_nx_1, sizeof(float),
										 sizeof(float), ny*nz,
										 cudaMemcpyDeviceToDevice));


			//checkCudaErrors(cudaMemcpy(h_work, d_work, 
			//						   nz*ny*nx*sizeof(float),
			//					       cudaMemcpyDeviceToHost));


			//for (k = 0; k < nz; k++){
			//	for (j = 0; j < ny; j++){
			//		i = j+k*ny;
			//		sendbuf[i] = h_work[k][j][nx-3];
			//	}
			//}

			//MPI_Send(sendbuf, ny*nz, MPI_FLOAT, 
			//		 dest_task, my_task, pom_comm);

			//MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
			//		 sour_task, sour_task, pom_comm, &status);

			//for (k = 0; k < nz; k++){
			//	for (j = 0; j < ny; j++){
			//		i = j+k*ny;
			//		h_work[k][j][nx-1] = recvbuf[i];
			//	}
			//}

			//for (k = 0; k < nz; k++){
			//	for (j = 0; j < ny; j++){
			//		i = j+k*ny;
			//		sendbuf[i] = h_work[k][j][nx-2];
			//	}
			//}
			//
			//MPI_Send(sendbuf, ny*nz, MPI_FLOAT,
			//		 dest_task, my_task, pom_comm);

			//checkCudaErrors(cudaMemcpy(d_work, h_work, 
			//						   nz*ny*nx*sizeof(float),
			//					       cudaMemcpyHostToDevice));

		}

		if (n_west == -1){

			float *west_send_2 =  d_2d_ny_nz_tmp0;
			float *west_recv_0 =  d_2d_ny_nz_tmp1;
			float *west_recv_1 =  d_2d_ny_nz_tmp2;

			checkCudaErrors(cudaMemcpy2D(west_send_2, sizeof(float),
									     d_work+2, nx*sizeof(float),
										 sizeof(float), ny*nz,
										 cudaMemcpyDeviceToDevice));

			sour_task = my_task+nproc_x-1;
			dest_task = my_task+nproc_x-1;

			MPI_Irecv(west_recv_0, ny*nz, MPI_FLOAT,
					 sour_task, 0, pom_comm, &request[0]);
			MPI_Irecv(west_recv_1, ny*nz, MPI_FLOAT,
					 sour_task, 1, pom_comm, &request[1]);
			MPI_Isend(west_send_2, ny*nz, MPI_FLOAT,
					 dest_task, 2, pom_comm, &request[2]);

			MPI_Waitall(3, request, status);


			checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
									     west_recv_0, sizeof(float),
										 sizeof(float), ny*nz,
										 cudaMemcpyDeviceToDevice));

			checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
									     west_recv_1, sizeof(float),
										 sizeof(float), ny*nz,
										 cudaMemcpyDeviceToDevice));


			//MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
			//		 sour_task, sour_task, pom_comm, &status);

			//for (k = 0; k < nz; k++){
			//	for (j = 0; j < ny; j++){
			//		i = j+k*ny;	
			//		h_work[k][j][0] = recvbuf[i];
			//	}
			//}

			//for (k = 0; k < nz; k++){
			//	for (j = 0; j < ny; j++){
			//		i = j+k*ny;	
			//		sendbuf[i] = h_work[k][j][2];
			//	}
			//}

			//MPI_Send(sendbuf, ny*nz, MPI_FLOAT,
			//		 dest_task, my_task, pom_comm);

			//MPI_Recv(recvbuf, ny*nz, MPI_FLOAT,
			//		 sour_task, sour_task, pom_comm, &status);

			//for (k = 0; k < nz; k++){
			//	for (j = 0; j < ny; j++){
			//		i = j+k*ny;	
			//		h_work[k][j][1] = recvbuf[i];
			//	}
			//}

			//checkCudaErrors(cudaMemcpy(d_work, h_work, 
			//						   nz*ny*nx*sizeof(float),
			//					       cudaMemcpyHostToDevice));
		}



	}
	
#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi3d_cuda_aware);
	xperi3d_cuda_aware_mpi_time += time_consumed(&time_start_xperi3d_cuda_aware,
							          &time_end_xperi3d_cuda_aware);
#endif

/*
      return
      end
*/
	return;

}


void xperi2d_cuda_aware_mpi(float *d_work, int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_aware,
				   time_end_xperi2d_cuda_aware;

	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_aware);
#endif

	int nproc_x;
	int dest_task, sour_task;

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


	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

	}else{
		MPI_Request request[3];
		MPI_Status status[3];

		if (n_east == -1){
			float *east_send_nx_2 = d_1d_ny_tmp0;
			float *east_send_nx_3 = d_1d_ny_tmp1;
			float *east_recv_nx_1 = d_1d_ny_tmp2;
			

			dest_task = my_task-nproc_x+1;	
			sour_task = my_task-nproc_x+1;

			checkCudaErrors(cudaMemcpy2D(east_send_nx_3, sizeof(float),
									     d_work+nx-3, nx*sizeof(float),
										 sizeof(float), ny,
										 cudaMemcpyDeviceToDevice));

			checkCudaErrors(cudaMemcpy2D(east_send_nx_2, sizeof(float),
									     d_work+nx-2, nx*sizeof(float),
										 sizeof(float), ny,
										 cudaMemcpyDeviceToDevice));

			MPI_Isend(east_send_nx_3, ny, MPI_FLOAT, 
					 dest_task, 0, pom_comm, &request[0]);
			MPI_Isend(east_send_nx_2, ny, MPI_FLOAT, 
					 dest_task, 1, pom_comm, &request[1]);
			MPI_Irecv(east_recv_nx_1, ny, MPI_FLOAT,
					 sour_task, 2, pom_comm, &request[2]);

			MPI_Waitall(3, request, status);

			checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
									     east_recv_nx_1, sizeof(float),
										 sizeof(float), ny,
										 cudaMemcpyDeviceToDevice));

			//for (j = 0; j < ny; j++){
			//	sendbuf[j] = h_work[j][nx-3];	
			//}

			//MPI_Send(sendbuf, ny, MPI_FLOAT, 
			//		 dest_task, my_task, pom_comm);
			//MPI_Recv(recvbuf, ny, MPI_FLOAT,
			//		 sour_task, sour_task, pom_comm, &status);

			//for (j = 0; j < ny; j++){
			//	h_work[j][nx-1] = recvbuf[j];	
			//}

			//for (j = 0; j < ny; j++){
			//	sendbuf[j] = h_work[j][nx-2];	
			//}
			//
			//MPI_Send(sendbuf, ny, MPI_FLOAT, 
			//		 dest_task, my_task, pom_comm);

			//checkCudaErrors(cudaMemcpy(d_work, h_work, ny*nx*sizeof(float),
			//					       cudaMemcpyHostToDevice));
		}

		if (n_west == -1){

			float *west_send_2 =  d_1d_ny_tmp0;
			float *west_recv_0 =  d_1d_ny_tmp1;
			float *west_recv_1 =  d_1d_ny_tmp2;

			sour_task = my_task+nproc_x-1;// ie. n_east == -1 and the same j
			dest_task = my_task+nproc_x-1;


			checkCudaErrors(cudaMemcpy2D(west_send_2, sizeof(float),
									     d_work+2, nx*sizeof(float),
										 sizeof(float), ny,
										 cudaMemcpyDeviceToDevice));

			MPI_Irecv(west_recv_0, ny, MPI_FLOAT, 
					  sour_task, 0, pom_comm, &request[0]);
			MPI_Irecv(west_recv_1, ny, MPI_FLOAT, 
					  sour_task, 1, pom_comm, &request[1]);
			MPI_Isend(west_send_2, ny, MPI_FLOAT, 
					 dest_task, 2, pom_comm, &request[2]);

			MPI_Waitall(3, request, status);

			checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
									     west_recv_0, sizeof(float),
										 sizeof(float), ny,
										 cudaMemcpyDeviceToDevice));

			checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
									     west_recv_1, sizeof(float),
										 sizeof(float), ny,
										 cudaMemcpyDeviceToDevice));


			//MPI_Recv(recvbuf, ny, MPI_FLOAT, 
			//		 sour_task, sour_task, pom_comm, &status);

			//for (j = 0; j < ny; j++){
			//	h_work[j][0] = recvbuf[j];		
			//}

			//for (j = 0; j < ny; j++){
			//	sendbuf[j] = h_work[j][2];	
			//}

			//MPI_Send(sendbuf, ny, MPI_FLOAT, 
			//		 dest_task, my_task, pom_comm);

			//MPI_Recv(recvbuf, ny, MPI_FLOAT, 
			//		 sour_task, sour_task, pom_comm, &status);

			//for (j = 0; j < ny; j++){
			//	h_work[j][1] = recvbuf[j];	
			//}

			//checkCudaErrors(cudaMemcpy(d_work, h_work, ny*nx*sizeof(float),
			//					       cudaMemcpyHostToDevice));
		}
	}

	/*
      return
      end
	*/

#ifndef TIME_DISABLE
	checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_aware);
	xperi2d_cuda_aware_mpi_time += time_consumed(&time_start_xperi2d_cuda_aware,
												 &time_end_xperi2d_cuda_aware);
#endif

	return;

}


void exchange2d_cuda_ipc(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_ipc,
				   time_end_exchange2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_ipc);
#endif

//! send ghost cell data to the east

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_send+1, nx*sizeof(float), 
		//							 d_west_recv+nx-1, nx*sizeof(float), 
		//							 sizeof(float), ny, 
		//							 cudaMemcpyDefault,
		//							 stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_west_recv+nx-1, nx*sizeof(float), 
									 d_send+1, nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){

	}

//! recieve ghost cell data from the south
	if (n_south != -1){

	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_ipc);
	exchange2d_cuda_ipc_time += 
					time_consumed(&time_start_exchange2d_cuda_ipc,
								  &time_end_exchange2d_cuda_ipc);
#endif

	return;
}

void send_east_cuda_ipc(float *d_send, 
						 float *d_east_recv, 
						 cudaStream_t &stream_in,
						 int nx, int ny){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_ipc,
				   time_end_exchange2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_ipc);
#endif

//! send ghost cell data to the east

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_ipc);
	exchange2d_cuda_ipc_time += 
					time_consumed(&time_start_exchange2d_cuda_ipc,
								  &time_end_exchange2d_cuda_ipc);
#endif

	return;
}

void exchange3d_cuda_ipc(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny, int nz){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_cuda_ipc,
				   time_end_exchange3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_cuda_ipc);
#endif

//! send ghost cell data to the east

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(d_west_recv+nx-1, nx*sizeof(float),
									 d_send+1, nx*sizeof(float), 
									 sizeof(float), ny*nz,
									 cudaMemcpyDefault,
									 stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){

	}

//! recieve ghost cell data from the south
	if (n_south != -1){

	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_cuda_ipc);
	exchange3d_cuda_ipc_time += 
					time_consumed(&time_start_exchange3d_cuda_ipc,
								  &time_end_exchange3d_cuda_ipc);
#endif

	return;
}


void xperi2d_cuda_ipc(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		MPI_Barrier(pom_comm);
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv, nx*sizeof(float),
								d_work+nx-3, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv+1, nx*sizeof(float),
								d_work+nx-2, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

		}

		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_east_most_recv+nx-1, nx*sizeof(float),
								d_work+2, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

		}
		checkCudaErrors(cudaStreamSynchronize(stream_in));
		MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif

	return;

}


void xperi3d_cuda_ipc(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi3d_cuda_ipc,
				   time_end_xperi3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi3d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		MPI_Barrier(pom_comm);
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv, nx*sizeof(float),
								d_work+nx-3, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv+1, nx*sizeof(float),
								d_work+nx-2, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

		}

		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_east_most_recv+nx-1, nx*sizeof(float),
								d_work+2, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

		}
		checkCudaErrors(cudaStreamSynchronize(stream_in));
		MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi3d_cuda_ipc);
	xperi3d_cuda_ipc_time += time_consumed(&time_start_xperi3d_cuda_ipc,
										   &time_end_xperi3d_cuda_ipc);
#endif

	return;

}


void exchange2d_cudaPeer(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 float *d_south_recv, 
						 float *d_north_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_ipc,
				   time_end_exchange2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_ipc);
#endif

	cudaMemcpy3DPeerParms p_east_recv={0};
	p_east_recv.extent = make_cudaExtent(sizeof(float), ny, 1);
	p_east_recv.dstDevice = n_east;
	p_east_recv.dstPtr = make_cudaPitchedPtr(d_east_recv, 
											 nx*sizeof(float), nx, ny);
	p_east_recv.srcDevice = my_task;
	p_east_recv.srcPtr = make_cudaPitchedPtr(d_send+(nx-2), 
											 nx*sizeof(float), nx, ny);

	cudaMemcpy3DPeerParms p_west_recv={0};
	p_west_recv.extent = make_cudaExtent(sizeof(float), ny, 1);
	p_west_recv.dstDevice = n_west;
	p_west_recv.dstPtr = make_cudaPitchedPtr(d_west_recv+(nx-1), 
											 nx*sizeof(float), nx, ny);
	p_west_recv.srcDevice = my_task;
	p_west_recv.srcPtr = make_cudaPitchedPtr(d_send+1, 
											 nx*sizeof(float), nx, ny);

//! send ghost cell data to the east
	if (n_east != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_east_recv, nx*sizeof(float), 
		//							 d_send+(nx-2), nx*sizeof(float), 
		//							 sizeof(float), ny, 
		//							 cudaMemcpyDefault,
		//							 stream_in));
		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_recv, stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_west_recv+nx-1, nx*sizeof(float), 
		//							 d_send+1, nx*sizeof(float), 
		//							 sizeof(float), ny, 
		//							 cudaMemcpyDefault,
		//							 stream_in));
		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_recv, stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		//checkCudaErrors(cudaMemcpyAsync(d_north_recv, 
		//								d_send+(ny-2)*nx, 
		//							    nx*sizeof(float), 
		//								cudaMemcpyDefault, stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		//checkCudaErrors(cudaMemcpyAsync(d_south_recv+(ny-1)*nx, 
		//								d_send+nx,
		//								nx*sizeof(float),
		//								cudaMemcpyDefault, stream_in));
	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_ipc);
	exchange2d_cuda_ipc_time += 
					time_consumed(&time_start_exchange2d_cuda_ipc,
								  &time_end_exchange2d_cuda_ipc);
#endif

	return;
}

void exchange2d_cudaPeerAsync(float *d_send, 
						      float *d_east_recv, 
						      float *d_west_recv,
						      float *d_south_recv, 
						      float *d_north_recv,
						      cudaStream_t &stream_in,
						      int nx, int ny){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_ipc,
				   time_end_exchange2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_ipc);
#endif

	cudaMemcpy3DPeerParms p_east_recv={0};
	p_east_recv.extent = make_cudaExtent(sizeof(float), ny, 1);
	p_east_recv.dstDevice = n_east;
	p_east_recv.dstPtr = make_cudaPitchedPtr(d_east_recv, 
											 nx*sizeof(float), nx, ny);
	p_east_recv.srcDevice = my_task;
	p_east_recv.srcPtr = make_cudaPitchedPtr(d_send+(nx-2), 
											 nx*sizeof(float), nx, ny);

	cudaMemcpy3DPeerParms p_west_recv={0};
	p_west_recv.extent = make_cudaExtent(sizeof(float), ny, 1);
	p_west_recv.dstDevice = n_west;
	p_west_recv.dstPtr = make_cudaPitchedPtr(d_west_recv+(nx-1), 
											 nx*sizeof(float), nx, ny);
	p_west_recv.srcDevice = my_task;
	p_west_recv.srcPtr = make_cudaPitchedPtr(d_send+1, 
											 nx*sizeof(float), nx, ny);

//! send ghost cell data to the east
	if (n_east != -1){
		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_recv, stream_in));
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_recv, stream_in));
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		//checkCudaErrors(cudaMemcpyAsync(d_north_recv, 
		//								d_send+(ny-2)*nx, 
		//							    nx*sizeof(float), 
		//								cudaMemcpyDefault, stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		//checkCudaErrors(cudaMemcpyAsync(d_south_recv+(ny-1)*nx, 
		//								d_send+nx,
		//								nx*sizeof(float),
		//								cudaMemcpyDefault, stream_in));
	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_ipc);
	exchange2d_cuda_ipc_time += 
					time_consumed(&time_start_exchange2d_cuda_ipc,
								  &time_end_exchange2d_cuda_ipc);
#endif

	return;
}

void exchange3d_cudaPeer(float *d_send, 
						 float *d_east_recv, 
						 float *d_west_recv,
						 float *d_south_recv, 
						 float *d_north_recv,
						 cudaStream_t &stream_in,
						 int nx, int ny, int nz){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_cuda_ipc,
				   time_end_exchange3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_cuda_ipc);
#endif

	cudaMemcpy3DPeerParms p_east_recv={0};
	p_east_recv.extent = make_cudaExtent(sizeof(float), ny, nz);
	p_east_recv.dstDevice = n_east;
	p_east_recv.dstPtr = make_cudaPitchedPtr(d_east_recv, 
											 nx*sizeof(float), nx, ny);
	p_east_recv.srcDevice = my_task;
	p_east_recv.srcPtr = make_cudaPitchedPtr(d_send+(nx-2), 
											 nx*sizeof(float), nx, ny);

	cudaMemcpy3DPeerParms p_west_recv={0};
	p_west_recv.extent = make_cudaExtent(sizeof(float), ny, nz);
	p_west_recv.dstDevice = n_west;
	p_west_recv.dstPtr = make_cudaPitchedPtr(d_west_recv+(nx-1), 
											 nx*sizeof(float), nx, ny);
	p_west_recv.srcDevice = my_task;
	p_west_recv.srcPtr = make_cudaPitchedPtr(d_send+1, 
											 nx*sizeof(float), nx, ny);


//! send ghost cell data to the east

	if (n_east != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_east_recv, nx*sizeof(float), 
		//							 d_send+(nx-2), nx*sizeof(float), 
		//							 sizeof(float), ny*nz, 
		//							 cudaMemcpyDefault,
		//							 stream_in));
		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_recv, stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_west_recv+nx-1, 
		//								  nx*sizeof(float),
		//							      d_send+1, nx*sizeof(float), 
		//								  sizeof(float), ny*nz,
		//							      cudaMemcpyDefault,
		//							      stream_in));

		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_recv, stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_north_recv, 
		//								  nx*ny*sizeof(float), 
		//								  d_send+(ny-2)*nx, 
		//								  nx*ny*sizeof(float), 
		//								  nx*sizeof(float), nz, 
		//								  cudaMemcpyDefault, stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_south_recv+(ny-1)*nx, 
		//								  nx*ny*sizeof(float), 
		//								  d_send+nx, 
		//								  nx*ny*sizeof(float), 
		//								  nx*sizeof(float), nz, 
		//								  cudaMemcpyDefault, stream_in));
	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_cuda_ipc);
	exchange3d_cuda_ipc_time += 
					time_consumed(&time_start_exchange3d_cuda_ipc,
								  &time_end_exchange3d_cuda_ipc);
#endif

	return;
}

void exchange3d_cudaPeerAsync(float *d_send, 
							  float *d_east_recv, 
							  float *d_west_recv,
							  float *d_south_recv, 
							  float *d_north_recv,
							  cudaStream_t &stream_in,
							  int nx, int ny, int nz){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_cuda_ipc,
				   time_end_exchange3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_cuda_ipc);
#endif

	cudaMemcpy3DPeerParms p_east_recv={0};
	p_east_recv.extent = make_cudaExtent(sizeof(float), ny, nz);
	p_east_recv.dstDevice = n_east;
	p_east_recv.dstPtr = make_cudaPitchedPtr(d_east_recv, 
											 nx*sizeof(float), nx, ny);
	p_east_recv.srcDevice = my_task;
	p_east_recv.srcPtr = make_cudaPitchedPtr(d_send+(nx-2), 
											 nx*sizeof(float), nx, ny);

	cudaMemcpy3DPeerParms p_west_recv={0};
	p_west_recv.extent = make_cudaExtent(sizeof(float), ny, nz);
	p_west_recv.dstDevice = n_west;
	p_west_recv.dstPtr = make_cudaPitchedPtr(d_west_recv+(nx-1), 
											 nx*sizeof(float), nx, ny);
	p_west_recv.srcDevice = my_task;
	p_west_recv.srcPtr = make_cudaPitchedPtr(d_send+1, 
											 nx*sizeof(float), nx, ny);


//! send ghost cell data to the east

	if (n_east != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_east_recv, nx*sizeof(float), 
		//							 d_send+(nx-2), nx*sizeof(float), 
		//							 sizeof(float), ny*nz, 
		//							 cudaMemcpyDefault,
		//							 stream_in));
		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_recv, stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_west_recv+nx-1, 
		//								  nx*sizeof(float),
		//							      d_send+1, nx*sizeof(float), 
		//								  sizeof(float), ny*nz,
		//							      cudaMemcpyDefault,
		//							      stream_in));

		checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_recv, stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_north_recv, 
		//								  nx*ny*sizeof(float), 
		//								  d_send+(ny-2)*nx, 
		//								  nx*ny*sizeof(float), 
		//								  nx*sizeof(float), nz, 
		//								  cudaMemcpyDefault, stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		//checkCudaErrors(cudaMemcpy2DAsync(d_south_recv+(ny-1)*nx, 
		//								  nx*ny*sizeof(float), 
		//								  d_send+nx, 
		//								  nx*ny*sizeof(float), 
		//								  nx*sizeof(float), nz, 
		//								  cudaMemcpyDefault, stream_in));
	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_cuda_ipc);
	exchange3d_cuda_ipc_time += 
					time_consumed(&time_start_exchange3d_cuda_ipc,
								  &time_end_exchange3d_cuda_ipc);
#endif

	return;
}

void exchange2d_cudaDHD(float *d_work, cudaStream_t &stream_in,
					int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_DHD,
				   time_end_exchange2d_DHD;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_DHD);
#endif

	int i, j, k;
	MPI_Request request[2];
	MPI_Status status[2];
	
	float *send_to_east = h_1d_ny_tmp0;
	float *recv_from_east = h_1d_ny_tmp1;
	float *send_to_west = h_1d_ny_tmp2;
	float *recv_from_west = h_1d_ny_tmp3;

	float *send_to_north = h_1d_nx_tmp0;
	float *recv_from_north = h_1d_nx_tmp1; 
	float *send_to_south = h_1d_nx_tmp2; 
	float *recv_from_south = h_1d_nx_tmp3; 
	

	if (n_east != -1){
		
		checkCudaErrors(cudaMemcpy2DAsync(send_to_east, sizeof(float), 
									      d_work+(nx-2), nx*sizeof(float), 
									      sizeof(float), ny, 
									      cudaMemcpyDeviceToHost,
									      stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		MPI_Isend(send_to_east, ny, MPI_FLOAT, n_east, my_task, pom_comm, &request[0]);
		MPI_Irecv(recv_from_east, ny, MPI_FLOAT, n_east, n_east, pom_comm, &request[1]);
	}

	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(send_to_west, sizeof(float), 
									      d_work+1, nx*sizeof(float), 
										  sizeof(float), ny, 
									      cudaMemcpyDeviceToHost,
									      stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		MPI_Irecv(recv_from_west, ny, MPI_FLOAT, n_west, n_west, pom_comm, &request[0]);	
		MPI_Isend(send_to_west, ny, MPI_FLOAT, n_west, my_task, pom_comm, &request[1]);
	}

	MPI_Waitall(2, request, status);

// send ghost cell data to the west
	if (n_west != -1){
		
		checkCudaErrors(cudaMemcpy2DAsync(d_work, nx*sizeof(float), 
									      recv_from_west, sizeof(float), 
									      sizeof(float), ny, 
									      cudaMemcpyHostToDevice,
										  stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(d_work+(nx-1), nx*sizeof(float), 
										  recv_from_east, sizeof(float), 
										  sizeof(float), ny, 
										  cudaMemcpyHostToDevice,
										  stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}

// send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpyAsync(send_to_north, 
										d_work+(ny-2)*nx, 
										nx*sizeof(float), 
										cudaMemcpyDeviceToHost,
										stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		MPI_Isend(send_to_north, nx, MPI_FLOAT, n_north, my_task, pom_comm, &request[0]);
		MPI_Irecv(recv_from_north, nx, MPI_FLOAT, n_north, n_north, pom_comm, &request[1]);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpyAsync(send_to_south, 
									    d_work+nx,
									    nx*sizeof(float),
									    cudaMemcpyDeviceToHost,
										stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));


		MPI_Irecv(recv_from_south, nx, MPI_FLOAT, n_south, n_south, pom_comm, &request[0]);
		MPI_Isend(send_to_south, nx, MPI_FLOAT, n_south, my_task, pom_comm, &request[1]);
	}
	MPI_Waitall(2, request, status);

// send ghost cell data to the south
	if (n_south != -1){

		checkCudaErrors(cudaMemcpyAsync(d_work,
									    recv_from_south,
									    nx*sizeof(float),
									    cudaMemcpyHostToDevice,
									    stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}

// recieve ghost cell data from the north
	if (n_north != -1){

		checkCudaErrors(cudaMemcpyAsync(d_work+(ny-1)*nx, 
									    recv_from_north,
									    nx*sizeof(float),
									    cudaMemcpyHostToDevice,
									    stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_DHD);
	exchange2d_cuda_aware_mpi_time += 
				time_consumed(&time_start_exchange2d_DHD,
				              &time_end_exchange2d_DHD);
#endif
	return;

}


void exchange3d_cudaDHD(float *d_work, cudaStream_t &stream_in,
					int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_DHDAsync,
				   time_end_exchange3d_DHDAsync;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_DHDAsync);
#endif

	int i, j, k;
	MPI_Request request[2];
	MPI_Status status[2];
	
	float *send_to_east = h_2d_ny_nz_tmp0;
	float *recv_from_east = h_2d_ny_nz_tmp1;
	float *send_to_west = h_2d_ny_nz_tmp2;
	float *recv_from_west = h_2d_ny_nz_tmp3;

	float *send_to_north = h_2d_nx_nz_tmp0;
	float *recv_from_north = h_2d_nx_nz_tmp1; 
	float *send_to_south = h_2d_nx_nz_tmp3; 
	float *recv_from_south = h_2d_nx_nz_tmp2; 
	

	if (n_east != -1){
		
		checkCudaErrors(cudaMemcpy2DAsync(send_to_east, sizeof(float), 
									      d_work+(nx-2), nx*sizeof(float), 
									      sizeof(float), ny*nz, 
									      cudaMemcpyDeviceToHost,
									      stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		MPI_Isend(send_to_east, ny*nz, MPI_FLOAT, n_east, my_task, pom_comm, &request[0]);
		MPI_Irecv(recv_from_east, ny*nz, MPI_FLOAT, n_east, n_east, pom_comm, &request[1]);
	}

	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(send_to_west, sizeof(float), 
									      d_work+1, nx*sizeof(float), 
										  sizeof(float), ny*nz, 
									      cudaMemcpyDeviceToHost,
									      stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		MPI_Irecv(recv_from_west, ny*nz, MPI_FLOAT, n_west, n_west, pom_comm, &request[0]);	
		MPI_Isend(send_to_west, ny*nz, MPI_FLOAT, n_west, my_task, pom_comm, &request[1]);
	}

	MPI_Waitall(2, request, status);

// send ghost cell data to the west
	if (n_west != -1){
		
		checkCudaErrors(cudaMemcpy2DAsync(d_work, nx*sizeof(float), 
									      recv_from_west, sizeof(float), 
									      sizeof(float), ny*nz, 
									      cudaMemcpyHostToDevice,
										  stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));
	}

// recieve ghost cell data from the east
	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(d_work+(nx-1), nx*sizeof(float), 
										  recv_from_east, sizeof(float), 
										  sizeof(float), ny*nz, 
										  cudaMemcpyHostToDevice,
										  stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}

// send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpy2DAsync(send_to_north, nx*sizeof(float), 
										  d_work+(ny-2)*nx, ny*nx*sizeof(float), 
										  nx*sizeof(float), nz, 
										  cudaMemcpyDeviceToHost,
										  stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

		MPI_Isend(send_to_north, nx*nz, MPI_FLOAT, n_north, my_task, pom_comm, &request[0]);
		MPI_Irecv(recv_from_north, nx*nz, MPI_FLOAT, n_north, n_north, pom_comm, &request[1]);
	}

// recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpy2DAsync(send_to_south, nx*sizeof(float), 
									      d_work+nx, ny*nx*sizeof(float), 
									      nx*sizeof(float), nz, 
									      cudaMemcpyDeviceToHost,
										  stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));


		MPI_Irecv(recv_from_south, nx*nz, MPI_FLOAT, n_south, n_south, pom_comm, &request[0]);
		MPI_Isend(send_to_south, nx*nz, MPI_FLOAT, n_south, my_task, pom_comm, &request[1]);
	}
	MPI_Waitall(2, request, status);

// send ghost cell data to the south
	if (n_south != -1){

		checkCudaErrors(cudaMemcpy2DAsync(d_work, ny*nx*sizeof(float), 
									      recv_from_south, nx*sizeof(float), 
									      nx*sizeof(float), nz, 
									      cudaMemcpyHostToDevice,
									      stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}

// recieve ghost cell data from the north
	if (n_north != -1){

		checkCudaErrors(cudaMemcpy2DAsync(d_work+(ny-1)*nx, ny*nx*sizeof(float), 
									      recv_from_north, nx*sizeof(float), 
									      nx*sizeof(float), nz, 
									      cudaMemcpyHostToDevice,
									      stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_DHDAsync);
	exchange3d_cuda_aware_mpi_time += time_consumed(&time_start_exchange3d_DHDAsync,
							             &time_end_exchange3d_DHDAsync);
#endif
	return;

}


void xperi2d_cudaPeer(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2DAsync(d_work+nx-1, nx*sizeof(float),
										  d_work+2, nx*sizeof(float),
										  sizeof(float), ny,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work, nx*sizeof(float),
										  d_work+nx-3, nx*sizeof(float),
										  sizeof(float), ny,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work+1, nx*sizeof(float),
										  d_work+nx-2, nx*sizeof(float),
										  sizeof(float), ny,
										  cudaMemcpyDeviceToDevice,
										  stream_in));
		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		MPI_Barrier(pom_comm);
		if (n_east == -1){

			int n_west_most=my_task-nproc_x+1;
			cudaMemcpy3DPeerParms p_west_most_recv0={0};
			p_west_most_recv0.extent = make_cudaExtent(sizeof(float), ny, 1);
			p_west_most_recv0.dstDevice = n_west_most;
			p_west_most_recv0.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv0.srcDevice = my_task;
			p_west_most_recv0.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-3), 
										  nx*sizeof(float), nx, ny);

			cudaMemcpy3DPeerParms p_west_most_recv1={0};
			p_west_most_recv1.extent = make_cudaExtent(sizeof(float), ny, 1);
			p_west_most_recv1.dstDevice = n_west_most;
			p_west_most_recv1.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv+1, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv1.srcDevice = my_task;
			p_west_most_recv1.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-2), 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv, nx*sizeof(float),
			//					d_work+nx-3, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv+1, nx*sizeof(float),
			//					d_work+nx-2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv0, 
												  stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv1, 
											      stream_in));
		}
		if (n_west == -1){

			int n_east_most=my_task+nproc_x-1;
			cudaMemcpy3DPeerParms p_east_most_recvnx1={0};
			p_east_most_recvnx1.extent 
					= make_cudaExtent(sizeof(float), ny, 1);
			p_east_most_recvnx1.dstDevice = n_east_most;
			p_east_most_recvnx1.dstPtr 
					= make_cudaPitchedPtr(d_east_most_recv+(nx-1), 
										  nx*sizeof(float), nx, ny);
			p_east_most_recvnx1.srcDevice = my_task;
			p_east_most_recvnx1.srcPtr 
					= make_cudaPitchedPtr(d_work+2, 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_east_most_recv+nx-1, nx*sizeof(float),
			//					d_work+2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_most_recvnx1, 
												  stream_in));

		}
		checkCudaErrors(cudaStreamSynchronize(stream_in));
		MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif
	return;
}

void xperi2d_cudaPeerAsync(float *d_work, 
						   float *d_east_most_recv,
						   float *d_west_most_recv,
						   cudaStream_t &stream_in,
						   int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2DAsync(d_work+nx-1, nx*sizeof(float),
										  d_work+2, nx*sizeof(float),
										  sizeof(float), ny,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work, nx*sizeof(float),
										  d_work+nx-3, nx*sizeof(float),
										  sizeof(float), ny,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work+1, nx*sizeof(float),
										  d_work+nx-2, nx*sizeof(float),
										  sizeof(float), ny,
										  cudaMemcpyDeviceToDevice,
										  stream_in));
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		//MPI_Barrier(pom_comm);
		if (n_east == -1){

			int n_west_most=my_task-nproc_x+1;
			cudaMemcpy3DPeerParms p_west_most_recv0={0};
			p_west_most_recv0.extent = make_cudaExtent(sizeof(float), ny, 1);
			p_west_most_recv0.dstDevice = n_west_most;
			p_west_most_recv0.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv0.srcDevice = my_task;
			p_west_most_recv0.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-3), 
										  nx*sizeof(float), nx, ny);

			cudaMemcpy3DPeerParms p_west_most_recv1={0};
			p_west_most_recv1.extent = make_cudaExtent(sizeof(float), ny, 1);
			p_west_most_recv1.dstDevice = n_west_most;
			p_west_most_recv1.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv+1, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv1.srcDevice = my_task;
			p_west_most_recv1.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-2), 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv, nx*sizeof(float),
			//					d_work+nx-3, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv+1, nx*sizeof(float),
			//					d_work+nx-2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv0, 
												  stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv1, 
											      stream_in));
		}
		if (n_west == -1){

			int n_east_most=my_task+nproc_x-1;
			cudaMemcpy3DPeerParms p_east_most_recvnx1={0};
			p_east_most_recvnx1.extent 
					= make_cudaExtent(sizeof(float), ny, 1);
			p_east_most_recvnx1.dstDevice = n_east_most;
			p_east_most_recvnx1.dstPtr 
					= make_cudaPitchedPtr(d_east_most_recv+(nx-1), 
										  nx*sizeof(float), nx, ny);
			p_east_most_recvnx1.srcDevice = my_task;
			p_east_most_recvnx1.srcPtr 
					= make_cudaPitchedPtr(d_work+2, 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_east_most_recv+nx-1, nx*sizeof(float),
			//					d_work+2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_most_recvnx1, 
												  stream_in));

		}
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
		//MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif

	return;

}

void xperi3d_cudaPeer(float *d_work, 
					  float *d_east_most_recv,
					  float *d_west_most_recv,
					  cudaStream_t &stream_in,
					  int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2DAsync(d_work+nx-1, nx*sizeof(float),
										  d_work+2, nx*sizeof(float),
										  sizeof(float), ny*nz,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work, nx*sizeof(float),
										  d_work+nx-3, nx*sizeof(float),
										  sizeof(float), ny*nz,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work+1, nx*sizeof(float),
										  d_work+nx-2, nx*sizeof(float),
										  sizeof(float), ny*nz,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaStreamSynchronize(stream_in));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		MPI_Barrier(pom_comm);
		if (n_east == -1){

			int n_west_most=my_task-nproc_x+1;
			cudaMemcpy3DPeerParms p_west_most_recv0={0};
			p_west_most_recv0.extent = make_cudaExtent(sizeof(float), ny, nz);
			p_west_most_recv0.dstDevice = n_west_most;
			p_west_most_recv0.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv0.srcDevice = my_task;
			p_west_most_recv0.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-3), 
										  nx*sizeof(float), nx, ny);

			cudaMemcpy3DPeerParms p_west_most_recv1={0};
			p_west_most_recv1.extent = make_cudaExtent(sizeof(float), ny, nz);
			p_west_most_recv1.dstDevice = n_west_most;
			p_west_most_recv1.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv+1, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv1.srcDevice = my_task;
			p_west_most_recv1.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-2), 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv, nx*sizeof(float),
			//					d_work+nx-3, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv+1, nx*sizeof(float),
			//					d_work+nx-2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv0, 
												  stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv1, 
											      stream_in));
		}
		if (n_west == -1){

			int n_east_most=my_task+nproc_x-1;
			cudaMemcpy3DPeerParms p_east_most_recvnx1={0};
			p_east_most_recvnx1.extent 
					= make_cudaExtent(sizeof(float), ny, nz);
			p_east_most_recvnx1.dstDevice = n_east_most;
			p_east_most_recvnx1.dstPtr 
					= make_cudaPitchedPtr(d_east_most_recv+(nx-1), 
										  nx*sizeof(float), nx, ny);
			p_east_most_recvnx1.srcDevice = my_task;
			p_east_most_recvnx1.srcPtr 
					= make_cudaPitchedPtr(d_work+2, 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_east_most_recv+nx-1, nx*sizeof(float),
			//					d_work+2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_most_recvnx1, 
												  stream_in));

		}
		checkCudaErrors(cudaStreamSynchronize(stream_in));
		MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif

	return;

}

void xperi3d_cudaPeerAsync(float *d_work, 
					       float *d_east_most_recv,
					       float *d_west_most_recv,
					       cudaStream_t &stream_in,
					       int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2DAsync(d_work+nx-1, nx*sizeof(float),
										  d_work+2, nx*sizeof(float),
										  sizeof(float), ny*nz,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work, nx*sizeof(float),
										  d_work+nx-3, nx*sizeof(float),
										  sizeof(float), ny*nz,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		checkCudaErrors(cudaMemcpy2DAsync(d_work+1, nx*sizeof(float),
										  d_work+nx-2, nx*sizeof(float),
										  sizeof(float), ny*nz,
										  cudaMemcpyDeviceToDevice,
										  stream_in));

		//checkCudaErrors(cudaStreamSynchronize(stream_in));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		//MPI_Barrier(pom_comm);
		if (n_east == -1){

			int n_west_most=my_task-nproc_x+1;
			cudaMemcpy3DPeerParms p_west_most_recv0={0};
			p_west_most_recv0.extent = make_cudaExtent(sizeof(float), ny, nz);
			p_west_most_recv0.dstDevice = n_west_most;
			p_west_most_recv0.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv0.srcDevice = my_task;
			p_west_most_recv0.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-3), 
										  nx*sizeof(float), nx, ny);

			cudaMemcpy3DPeerParms p_west_most_recv1={0};
			p_west_most_recv1.extent = make_cudaExtent(sizeof(float), ny, nz);
			p_west_most_recv1.dstDevice = n_west_most;
			p_west_most_recv1.dstPtr 
					= make_cudaPitchedPtr(d_west_most_recv+1, 
										  nx*sizeof(float), nx, ny);
			p_west_most_recv1.srcDevice = my_task;
			p_west_most_recv1.srcPtr 
					= make_cudaPitchedPtr(d_work+(nx-2), 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv, nx*sizeof(float),
			//					d_work+nx-3, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_west_most_recv+1, nx*sizeof(float),
			//					d_work+nx-2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv0, 
												  stream_in));
			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_west_most_recv1, 
											      stream_in));
		}
		if (n_west == -1){

			int n_east_most=my_task+nproc_x-1;
			cudaMemcpy3DPeerParms p_east_most_recvnx1={0};
			p_east_most_recvnx1.extent 
					= make_cudaExtent(sizeof(float), ny, nz);
			p_east_most_recvnx1.dstDevice = n_east_most;
			p_east_most_recvnx1.dstPtr 
					= make_cudaPitchedPtr(d_east_most_recv+(nx-1), 
										  nx*sizeof(float), nx, ny);
			p_east_most_recvnx1.srcDevice = my_task;
			p_east_most_recvnx1.srcPtr 
					= make_cudaPitchedPtr(d_work+2, 
										  nx*sizeof(float), nx, ny);

			//checkCudaErrors(cudaMemcpy2DAsync(
			//					d_east_most_recv+nx-1, nx*sizeof(float),
			//					d_work+2, nx*sizeof(float),
			//					sizeof(float), ny,
			//					cudaMemcpyDefault,
			//					stream_in));

			checkCudaErrors(cudaMemcpy3DPeerAsync(&p_east_most_recvnx1, 
												  stream_in));

		}
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
		//MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif

	return;

}

void exchange2d_cudaUVA(float *d_send, 
						float *d_east_recv, 
						float *d_west_recv,
						float *d_south_recv, 
						float *d_north_recv,
						cudaStream_t &stream_in,
						int nx, int ny){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_ipc,
				   time_end_exchange2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_ipc);
#endif

//! send ghost cell data to the east
	MPI_Barrier(pom_comm);

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_west_recv+nx-1, nx*sizeof(float), 
									 d_send+1, nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

	checkCudaErrors(cudaStreamSynchronize(stream_in));
	MPI_Barrier(pom_comm);

//! send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpyAsync(d_north_recv, 
									    d_send+(ny-2)*nx, 
									    nx*sizeof(float),
									    cudaMemcpyDefault,
									    stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpyAsync(d_south_recv+(ny-1)*nx, 
									    d_send+nx, 
									    nx*sizeof(float), 
									    cudaMemcpyDefault,
									    stream_in));
	}

	checkCudaErrors(cudaStreamSynchronize(stream_in));
	MPI_Barrier(pom_comm);

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_ipc);
	exchange2d_cuda_ipc_time += 
					time_consumed(&time_start_exchange2d_cuda_ipc,
								  &time_end_exchange2d_cuda_ipc);
#endif

	return;
}

/*
void exchange2d_cudaUVAAsync(float *d_send, 
						     float *d_east_recv, 
						     float *d_west_recv,
						     float *d_south_recv, 
						     float *d_north_recv,
						     cudaStream_t &stream_in,
						     int nx, int ny){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange2d_cuda_ipc,
				   time_end_exchange2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange2d_cuda_ipc);
#endif

//! send ghost cell data to the east
	//MPI_Barrier(pom_comm);

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_west_recv+nx-1, nx*sizeof(float), 
									 d_send+1, nx*sizeof(float), 
									 sizeof(float), ny, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpyAsync(d_north_recv, 
									    d_send+(ny-2)*nx, 
									    nx*sizeof(float),
									    cudaMemcpyDefault,
									    stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpyAsync(d_south_recv+(ny-1)*nx, 
									    d_send+nx, 
									    nx*sizeof(float), 
									    cudaMemcpyDefault,
									    stream_in));
	}

	//checkCudaErrors(cudaStreamSynchronize(stream_in));
	//MPI_Barrier(pom_comm);

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange2d_cuda_ipc);
	exchange2d_cuda_ipc_time += 
					time_consumed(&time_start_exchange2d_cuda_ipc,
								  &time_end_exchange2d_cuda_ipc);
#endif

	return;
}
*/

void exchange3d_cudaUVA(float *d_send, 
						float *d_east_recv, 
						float *d_west_recv,
						float *d_south_recv,
						float *d_north_recv, 
						cudaStream_t &stream_in,
						int nx, int ny, int nz){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_cuda_ipc,
				   time_end_exchange3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_cuda_ipc);
#endif

//! send ghost cell data to the east

	MPI_Barrier(pom_comm);

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_west_recv+nx-1, nx*sizeof(float),
									 d_send+1, nx*sizeof(float), 
									 sizeof(float), ny*nz,
									 cudaMemcpyDefault,
									 stream_in));
	}

	checkCudaErrors(cudaStreamSynchronize(stream_in));
	MPI_Barrier(pom_comm);

//! send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_north_recv, ny*nx*sizeof(float), 
									 d_send+(ny-2)*nx, ny*nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_south_recv+(ny-1)*nx, ny*nx*sizeof(float), 
									 d_send+nx, ny*nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

	checkCudaErrors(cudaStreamSynchronize(stream_in));
	MPI_Barrier(pom_comm);

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_cuda_ipc);
	exchange3d_cuda_ipc_time += 
					time_consumed(&time_start_exchange3d_cuda_ipc,
								  &time_end_exchange3d_cuda_ipc);
#endif

	return;
}


/*
void exchange3d_cudaUVAAsync(float *d_send, 
						     float *d_east_recv, 
						     float *d_west_recv,
						     float *d_south_recv,
						     float *d_north_recv, 
						     cudaStream_t &stream_in,
						     int nx, int ny, int nz){
#ifndef TIME_DISABLE
	struct timeval time_start_exchange3d_cuda_ipc,
				   time_end_exchange3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_exchange3d_cuda_ipc);
#endif

//! send ghost cell data to the east

	//MPI_Barrier(pom_comm);

	if (n_east != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_east_recv, nx*sizeof(float), 
									 d_send+(nx-2), nx*sizeof(float), 
									 sizeof(float), ny*nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the west
	if (n_west != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_west_recv+nx-1, nx*sizeof(float),
									 d_send+1, nx*sizeof(float), 
									 sizeof(float), ny*nz,
									 cudaMemcpyDefault,
									 stream_in));
	}

//! send ghost cell data to the north
	if (n_north != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_north_recv, ny*nx*sizeof(float), 
									 d_send+(ny-2)*nx, ny*nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

//! recieve ghost cell data from the south
	if (n_south != -1){
		checkCudaErrors(cudaMemcpy2DAsync(
									 d_south_recv+(ny-1)*nx, ny*nx*sizeof(float), 
									 d_send+nx, ny*nx*sizeof(float), 
									 nx*sizeof(float), nz, 
									 cudaMemcpyDefault,
									 stream_in));
	}

	//checkCudaErrors(cudaStreamSynchronize(stream_in));
	//MPI_Barrier(pom_comm);

#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_exchange3d_cuda_ipc);
	exchange3d_cuda_ipc_time += 
					time_consumed(&time_start_exchange3d_cuda_ipc,
								  &time_end_exchange3d_cuda_ipc);
#endif

	return;
}
*/

void xperi2d_cudaUVA(float *d_work, 
					 float *d_east_most_recv,
					 float *d_west_most_recv,
					 cudaStream_t &stream_in,
					 int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		MPI_Barrier(pom_comm);
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv, nx*sizeof(float),
								d_work+nx-3, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv+1, nx*sizeof(float),
								d_work+nx-2, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

		}

		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_east_most_recv+nx-1, nx*sizeof(float),
								d_work+2, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

		}
		checkCudaErrors(cudaStreamSynchronize(stream_in));
		MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif

	return;

}

void xperi2d_cudaUVAAsync(float *d_work, 
						  float *d_east_most_recv,
						  float *d_west_most_recv,
					      cudaStream_t &stream_in,
					      int nx, int ny){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi2d_cuda_ipc,
				   time_end_xperi2d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi2d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny,
									 cudaMemcpyDeviceToDevice));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		//MPI_Barrier(pom_comm);
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv, nx*sizeof(float),
								d_work+nx-3, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv+1, nx*sizeof(float),
								d_work+nx-2, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

		}

		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_east_most_recv+nx-1, nx*sizeof(float),
								d_work+2, nx*sizeof(float),
								sizeof(float), ny,
								cudaMemcpyDefault,
								stream_in));

		}
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
		//MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi2d_cuda_ipc);
	xperi2d_cuda_ipc_time += time_consumed(&time_start_xperi2d_cuda_ipc,
										   &time_end_xperi2d_cuda_ipc);
#endif

	return;

}


void xperi3d_cudaUVA(float *d_work, 
					 float *d_east_most_recv,
					 float *d_west_most_recv,
					 cudaStream_t &stream_in,
					 int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi3d_cuda_ipc,
				   time_end_xperi3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi3d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		MPI_Barrier(pom_comm);
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv, nx*sizeof(float),
								d_work+nx-3, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv+1, nx*sizeof(float),
								d_work+nx-2, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

		}

		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_east_most_recv+nx-1, nx*sizeof(float),
								d_work+2, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

		}
		checkCudaErrors(cudaStreamSynchronize(stream_in));
		MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi3d_cuda_ipc);
	xperi3d_cuda_ipc_time += time_consumed(&time_start_xperi3d_cuda_ipc,
										   &time_end_xperi3d_cuda_ipc);
#endif

	return;

}


void xperi3d_cudaUVAAsync(float *d_work, 
					      float *d_east_most_recv,
					      float *d_west_most_recv,
					      cudaStream_t &stream_in,
					      int nx, int ny, int nz){

#ifndef TIME_DISABLE
	struct timeval time_start_xperi3d_cuda_ipc,
				   time_end_xperi3d_cuda_ipc;

	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_start_xperi3d_cuda_ipc);
#endif

	int nproc_x;
	if ((im_global-2) % (im_local-2) == 0){
		nproc_x = (im_global-2)/(im_local-2);
	}else{
		nproc_x = (im_global-2)/(im_local-2)+1;
	}

	if (nproc_x == 1){
		/*
		for (j = 0; j < ny; j++){
			work[j][nx-1] = work[j][2];	
			work[j][0] = work[j][nx-3];
			work[j][1] = work[j][nx-2];
		}
		*/
		checkCudaErrors(cudaMemcpy2D(d_work+nx-1, nx*sizeof(float),
								     d_work+2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work, nx*sizeof(float),
								     d_work+nx-3, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

		checkCudaErrors(cudaMemcpy2D(d_work+1, nx*sizeof(float),
								     d_work+nx-2, nx*sizeof(float),
									 sizeof(float), ny*nz,
									 cudaMemcpyDeviceToDevice));

	}else{
		//printf("rank %d: n_east=%d, n_west=%d\n", my_task, n_east, n_west);
		//MPI_Barrier(pom_comm);
		if (n_east == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv, nx*sizeof(float),
								d_work+nx-3, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

			checkCudaErrors(cudaMemcpy2DAsync(
								d_west_most_recv+1, nx*sizeof(float),
								d_work+nx-2, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

		}

		if (n_west == -1){
			checkCudaErrors(cudaMemcpy2DAsync(
								d_east_most_recv+nx-1, nx*sizeof(float),
								d_work+2, nx*sizeof(float),
								sizeof(float), ny*nz,
								cudaMemcpyDefault,
								stream_in));

		}
		//checkCudaErrors(cudaStreamSynchronize(stream_in));
		//MPI_Barrier(pom_comm);
	}


#ifndef TIME_DISABLE
	//checkCudaErrors(cudaDeviceSynchronize());
	timer_now(&time_end_xperi3d_cuda_ipc);
	xperi3d_cuda_ipc_time += time_consumed(&time_start_xperi3d_cuda_ipc,
										   &time_end_xperi3d_cuda_ipc);
#endif

	return;

}
