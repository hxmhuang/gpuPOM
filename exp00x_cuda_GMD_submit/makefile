#
# sbPOM makefile
#

#-----------------------------------------------------------------------
# Settings that depend on the system and the compiler
#-----------------------------------------------------------------------
# Set macros
CPP = cpp -P
#FC = ftn   !yoyo 
#LD = ftn
FC = mpiifort
CC = mpiicc
NVC = nvcc

LD = mpiicc
CLEAN = rm
# Set libraries and include files
# NETCDFINC = -I/autofs/na1_sw/eslogin/p-netcdf/1.1.1/cnl3.1.gnu/include  # yoyo
# NETCDFLIB = -L/autofs/na1_sw/eslogin/p-netcdf/1.1.1/cnl3.1.gnu/lib
NETCDFINC = -I/home/pom_test/software/include
NETCDFLIB = -L/home/pom_test/software/lib
#NETCDFINC = -I/home/pom_test/netcdf_ompi_compatible/include
#NETCDFLIB = -L/home/pom_test/netcdf_ompi_compatible/lib
FORTRANLIB = -L/home/pom_test/intel/composerxe/lib/intel64 -lifcore -lifport

#FFLAGS = -ip -O3 -lmpi  $(NETCDFINC)
#FFLAGS = -check bounds -check format -lmpi  $(NETCDFINC)
#FFLAGS = -check bounds  -lmpi  $(NETCDFINC)
#FFLAGS = -O3 -fp-model precise  -assume byterecl $(NETCDFINC)
FFLAGS = -O3 -fp-model precise  $(NETCDFINC)
CFLAGS = -O3 -fp-model precise  $(NETCDFINC) 
NVCFLAGS = -m64 -O3 \
		   -lineinfo --ptxas-options=-v -keep\
		   -gencode arch=compute_35,code=sm_35\
 		   -I/home/pom_test/intel/impi/4.1.3.048/include64\
		   -DMPICH_IGNORE_CXX_SEEK\
		   -DPURE_GPU_MPI \
		   --fmad=false\
		   -DTEST_LOOP_FUSION 
# -I/opt/intel/impi/4.1.1/include64\
# -I/home/pom/intel/impi/4.1.1.036/include64/\
# -I/home/pom_test/openmpi/include\
# -I/home/pom_test/openmpi/include\

#FFLAGS = -O3 -debug minimal -fp-model precise -override-limits $(NETCDFINC)

#LIBS = -ip  -O3  -lmpi $(NETCDFLIB) -lpnetcdf 
#LIBS =  -check bounds  -lmpi $(NETCDFLIB) -lpnetcdf 
LIBS = -O3 -fp-model precise $(NETCDFLIB) -lpnetcdf $(FORTRANLIB) \
	   -L/usr/local/cuda-5.5/lib64 -lcudart 

# LIBS = -O3 -debug minimal -fp-model precise -override-limits $(NETCDFLIB) -lpnetcdf
#-----------------------------------------------------------------------
# Set the executable
#-----------------------------------------------------------------------
BIN = pom.exe  #yoyo


#-----------------------------------------------------------------------
# Define source directory
#-----------------------------------------------------------------------
SRCDIR = pom

#-----------------------------------------------------------------------
# Define objects
#-----------------------------------------------------------------------
OBJS = csolver_gpu.o   \
	   cparallel_mpi.o  \
       cparallel_mpi_gpu.o \
       data.o     		\
	   cadvance_gpu.o   \
	   cu_data.o   \
	   cu_data_io.o   \
       cbounds_forcing.o \
       cbounds_forcing_gpu.o \
	   read_namelist.o  \
       cinitialize.o     \
	   timer_all.o      \
       io_pnetcdf.o     \
       cio_pnetcdf.o     \
       cadvance.o        \
       csolver.o         \
	   timer_all.o      \
	   cpom.o            \
       cwind.o          \
       criver.o         \
       cassim.o         \
       ctsforce.o       \
       cmcsst.o         \
       cassim_drf.o     \
       ctrajdrf.o       \
       cinterp.o          

VPATH = $(SRCDIR)

#-----------------------------------------------------------------------
# Set implicit rules for compilation
#-----------------------------------------------------------------------
%.o: %.f 
	@echo
	$(FC) -c $(FFLAGS)  $<

%.o: %.f 
	@echo
	$(FC) -c $(FFLAGS)  $<

%.o: %.c 
	@echo
	$(CC) -c $(CFLAGS)  $<

%.o: %.cu
	@echo
	$(NVC) -c $(NVCFLAGS)  $<

#-----------------------------------------------------------------------
# Set implicit rules for dependencies
#-----------------------------------------------------------------------
%.f: %.F
	@echo
	$(CPP) $(FFLAGS) $< > $*.f

#%.o:%.f90 
#	@echo
#	$(FC) -c $(FFLAGS)  $<
#-----------------------------------------------------------------------
# Create the executable
#-----------------------------------------------------------------------
$(BIN): $(OBJS)
	@echo
	$(LD) $(FFLAGS) -o $(BIN) $(OBJS) $(LIBS)

#-----------------------------------------------------------------------
# Cleaning target
#-----------------------------------------------------------------------
clean:
	@rm -f *.o *.mod *.il *.exe 
