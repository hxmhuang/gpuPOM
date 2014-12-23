#!/bin/sh

#This is the runscript for GPU_POM
#Set up for idealized tests grid=-1 (standing wave) or grid=-2 (dam-break)
#You can learn how to set up your own test cases by searching for "ngrid"
#in initialize.f and wind.f
#
#The model produces *.nc files which can be viewed using freeware ncview
#
#---------------------------------------------------------------------------
#To run, change:
#1. homdir = home directory where batch*, make* pom.h & pom/*.f etc are:
#2. inpdir = directory where input data, e.g. in/gfsw, in/tsclim, in/ssts etc
#3. rundir = directory where this job is run
#4. "date_start0","date_end0", & "days"
#5. "num_of_nodes" should be consistent with n_proc in pom.h
#6. "run_time"
#---------------------------------------------------------------------------
#Reminder of things that may also have to be changed depending on your run:
#a. days
#b. write_rst
#c. prtd1 & prtd2
#---------------------------------------------------------------------------
# Also search for '#?' below
#---------------------------------------------------------------------------
#
#set title etc and start & end dates for this experiment (e.g. can be 50years)

debug=1

im_global=202
jm_global=47
kb=3
im_local=52
jm_local=47
im_global_coarse=202
jm_global_coarse=47
im_local_coarse=52
jm_local_coarse=47
x_division=2
y_division=2
n_proc=4


date_startfix="1987-07-02" #?reference date: all dates are referenced to this
date_start0="2012-01-17"   #?start date, it is arbitrary for test cases grid<0
days=14                    #?simulation days from date_start0
date_end0=`date -d "$date_start0 +$days days" +%Y-%m-%d`


title="testsexp001"        #?run title
windf="gfsw"               #?type of wind, used only for realistic case
num_of_nodes=4
run_time="99:00:00"


#flags (set=0 to turn off, e.g. tide=0 etc):usually=0 for idealized case
grid=-1					   #?<=0 to specify grid directly in initialize.f
						   #?Then set all but vort below (i.e. tsforce ..) = 0
						   #?=-1 topograhic standing wave in an f-plane channel
						   #?=-2 dambreak (warm_S-cold_N) in f-plane zonal channel

tsforce=0
wind=0
river=0
assimssh=0
assimsst=0
assimdrf=0
tide=0
trajdrf=0
stokes=0
vort=1                     #?vorticity analysis useful in idealized run also
rst_flag=0				   #?=0 1st run, =1 subsequent runs to read restart
						   #?file from a previous run

if [ $grid -lt 0 ]; then
	mgrid=`echo "-1*$grid" | bc`
	if [ $mgrid -lt 10 ]; then
		title="testsexp00"$mgrid
	fi

	if [ $mgrid -lt 100 ] && [ $mgrid -gt 9 ]; then
		title="testsexp0"$mgrid
    fi	
fi


#input data files (e.g. in/gfsw, in/tsclim, in/ssts etc):
#Not used if grid, wind, assimssh & assimsst above =0

##below are not used.. commented by xsz
if [ $grid -eq 1 ]; then    #?for run w/realistic coastline & topography etc
	inpdir='/archive4/lyo/gridf/pac10-2/out_fracetopo20_SLMIN0.1_slmax1.0_landfill_trench'
	inpdirgfsw='/archive4/hunglu/Oceanus/cron_gfs_download_interp/out/'
	inpdirssha='/archive4/lameixisi/AVISO/for_netcdf/nc2sbpom/out/'
	inpdircorr='/archive4/lameixisi/bin2net/tmp/'
	inpdirmcsst='/archive4/EnvironData/MCSST/pac10/'

	#fakerest is useful for restart from some different restart file (to test etc)
	fakerest='/archive8/lyo/out/restart.1990-01-17_00_00_00.nc'
else                           #?for idealized runs
	inpdir='/NotUsed/'
	inpdirgfsw='/NotUsed/'
	inpdirssha='/NotUsed/'
	inpdircorr='/NotUsed/'
	inpdirmcsst='/NotUsed/'
	fakerest='/NotUsed/'
fi
##end.. commented by xsz


if [ $grid -eq 1 ]; then
	griddir=$inpdir'/in/grid'
fi

if [ $tsforce -eq 1 ]; then
	tsclimdir=$inpdir'/in/tsclim'
	sstsdir=$inpdir'/in/ssts'
fi

if [ $wind -eq 1 ]; then
	winddir=$inpdirgfsw
fi

if [ $river -eq 1 ]; then
	riverdir=$inpdir'/in/river'
fi

if [ $assimssh -eq 1 ]; then
	assimsshdir=$inpdirssha
	assimcordir=$inpdircorr
fi

if [ $assimsst -eq 1 ]; then
	assimsstdir=$inpdirmcsst
fi

if [ $assimdrf -eq 1 ]; then
	assimdrfdir=$inpdir'/in/assimdrf'
fi

if [ $tide -eq 1 ]; then
	tidedir=$inpdir'/in/tide'
fi


#home directory where batch*, make* pom.h & pom/*.f etc are:
homdir='/home/pom/mpi_pom/leo/tests/exp00x_cuda_debug'

#run directory (where we are running this simulation, usually a scratch disk):
rundir="/home/pom/mpi_pom/leo/tests/exp00x_cuda_debug/run"
if [ ! -d $rundir ]; then
	mkdir $rundir
fi
 
if [ $trajdrf -eq 1 ]; then
	mkdir -p $rundir/out/trajdrf 
fi
#
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#
date_start=$date_start0
date_end=$date_start0
#
# --------------------------------------------------------------
# link codes and input files from homdir to rundir
# --------------------------------------------------------------
cd $rundir
echo "cd $rundir"

rm -f pom.h pom/*

# --------------------------------------------------------------
#?The following section is for realistic run -------------------
#Comment out:renew links every run for new sat data
#if [-e $rundir/in ]; then 
#    echo "$rundir/in/ directory exists! "
#else        #Renew links every run for new satellite data

if [ -d $rundir/in ]; then 
    echo "********** run/in/ directory exist ********** "
else
    echo "********** Create, link and copy etc ******** "
    mkdir in
fi

    if [ $grid -eq 1 ]; then
		tempo=in/grid
		ln -s $griddir $tempo 
		echo "Done copying grid to "$rundir/$tempo
    fi 

    if [ $tsforce -eq 1 ]; then
		tempo=in/tsclim
		ln -s $tsclimdir $tempo 
		echo "Done copying tsclim to "$rundir/$tempo
		set tempo=in/ssts
		ln -s $sstsdir $tempo 
		echo "Done copying ssts to "$rundir/$tempo
    fi 

    if [ $wind -eq 1 ]; then
		#set tempo=in/gfsw
		tempo=in/$windf
		ln -s $winddir $tempo 
		echo "Done copying wind to "$rundir/$tempo
    fi 

    if [ $river -eq 1 ]; then
		#set tempo=in/river
		#mkdir $tempo
		cp $riverdir/*riv*.bin in/.
		echo "Done copying river to "$rundir"/in"
    fi 

    if [ $assimssh -eq 1 ]; then
		tempo=in/assim
		mkdir $tempo
		ln -s $assimsshdir/msla_????????.nc $tempo/.
		ln -s $assimcordir/assiminfo.nc $tempo/.
		echo "Done copying assimssh to "$rundir/$tempo
    fi 

    if [ $assimsst -eq 1 ]; then
		tempo=in/mcsst
		mkdir $tempo
		ln -s $assimsstdir/mcsst_????????.nc $tempo/.
		echo "Done copying assimsst to "$rundir/$tempo
    fi 

    if [ $assimdrf -eq 1 ]; then
		tempo=in/assimdrf
		mkdir $tempo
		cp $assimdrfdir/uvlatlon?????????? $tempo/.
		echo "Done copying assimdrf to "$rundir/$tempo
    fi 

    if [ $tide -eq 1 ]; then
#   set tempo=in/tide
#   mkdir $tempo
		cp $tidedir/tide.nc in/.
		echo "Done copying tide to "$rundir"/in"
    fi

#endif   #Comment out: renew links every run for new sat data
#?The above section is for realistic run -----------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# compile
# --------------------------------------------------------------

echo "Linking files from $homdir to $rundir ..."
#Load atop aravision (computer-cluster-specific)
#module load intel_poe ??what does this line do?

make clean
rm -f $rundir/batch_run_profs-fcast-var.csh

ln -sf $homdir/makefile .
if [ ! -d $rundir/out ]; then
	mkdir $rundir/out
fi

cp $homdir/run_exp001.sh $rundir

if [ -e $rundir/pom.exe ]; then
	make
else
	cp -r $homdir/pom $rundir/.
	cp $homdir/pom.h .


	sourceFile=$rundir/pom/utils.h
	sourceFileF=$rundir/pom.h
	if [ ! -e $sourceFile ]; then 
		echo $sourceFile does not exist
	fi

    sed -i "s/#define i_global_size [0-9]\+/#define i_global_size \
			$im_global/" $sourceFile
    sed -i "s/#define j_global_size [0-9]\+/#define j_global_size \
			$jm_global/" $sourceFile
    sed -i "s/#define k_size [0-9]\+/#define k_size $kb/" $sourceFile
    sed -i "s/#define j_size [0-9]\+/#define j_size $jm_local/" $sourceFile
    sed -i "s/#define j_coarse_size [0-9]\+/#define j_coarse_size \
		    $jm_local_coarse/" $sourceFile
    sed -i "s/#define i_size [0-9]\+/#define i_size $im_local/" $sourceFile
    sed -i "s/#define i_coarse_size [0-9]\+/#define i_coarse_size \
		    $im_local_coarse/" $sourceFile
    sed -i "s/kb=[0-9]\+/kb=$kb/" $sourceFileF

	make
fi


if [ $? -ne 0 ]; then exit; fi

echo "Done!"
echo " "
echo "Compling Done!"
echo " "

# ---------------------------------------------------------------
# checking running days and date_end
# ---------------------------------------------------------------
if [ "$date_end" != "$date_end0" ]; then
	date_end=`date -d "$date_start +$days days" +%Y-%m-%d `
	sec1=`date -d $date_end +%s `
    sec2=`date -d $date_end0 +%s `
	if [ $sec1 -gt $sec2 ]; then
		sec1=`date -d $date_start '+%s '`
		days=$[($sec2-$sec1)/86400]
		date_end=$date_end0
	fi
fi


#----------------------------------------------------------------
#                   creating pom.nml
#Model parameters which may need to be changed for specific problems
# ---------------------------------------------------------------
# pom_nml -------------------------------------------------------
namelist_file="pom.nml."$date_start

if [ $debug -eq 1 ]; then
	echo "it is in debug mode"
	namelist_file_debug="pom_debug.nml."$date_start
fi

restart_file="restart."$date_start"_00_00_00.nc"
netcdf_file="'$title"."$date_start"_to_"$date_end'"
mode=3
nadv=1
nitera=2
sw=0.5
npg=1				#?pressure gradient scheme  1->2nd order; 2->4th order
dte=10.
isplit=45
prtd1=$days			#?interval of 3D outputs
prtd2=1.0			#?interval of 2D (surface) outputs
iperx=0
ipery=0
n1d=0

#1st run has no restart, but others satisfying the following "if's" have
if [ "$date_start" != "$date_start0" ]; then 
	rst_flag=1
fi

if [ -e $rundir/out/restart."$date_start"_00_00_00.nc ]; then
	rst_flag=1
else
	if [ -e $inpdir/out/restart."$date_start"_00_00_00.nc ]; then
		cp $inpdir/out/restart."$date_start"_00_00_00.nc $rundir/out/.
		rst_flag=1
	fi

	#Use faked restart?
	if [ -e $fakerest ]; then
		ln -s $fakerest $rundir/out/restart."$date_start"_00_00_00.nc
		rst_flag=1
	fi
fi

nread_rst=$rst_flag
#write_rst = $days
write_rst=99999
if [ $days -lt $write_rst ]; then
	write_rst=$days
fi

write_rst_file='restart'

echo "& pom_nml "								>  $namelist_file
echo " title = '"$title"'"						>> $namelist_file
echo " netcdf_file = " $netcdf_file				>> $namelist_file
echo " mode = " $mode                			>> $namelist_file
echo " nadv = " $nadv                			>> $namelist_file
echo " nitera = " $nitera            			>> $namelist_file
echo " sw = " $sw                    			>> $namelist_file
echo " npg = " $npg                  			>> $namelist_file
echo " dte = " $dte                  			>> $namelist_file
echo " isplit = " $isplit            			>> $namelist_file
echo " time_start ='"$date_startfix"_00:00:00'" >> $namelist_file
echo " nread_rst = " $nread_rst					>> $namelist_file
echo " read_rst_file = '"$restart_file"'"		>> $namelist_file
echo " write_rst = " $write_rst					>> $namelist_file
echo " write_rst_file = " $write_rst_file		>> $namelist_file
echo " days = " $days							>> $namelist_file
echo " prtd1 = " $prtd1              			>> $namelist_file
echo " prtd2 = " $prtd2              			>> $namelist_file
echo " iperx = " $iperx              			>> $namelist_file
echo " ipery = " $ipery              			>> $namelist_file
echo " n1d = "   $n1d                			>> $namelist_file
echo " ngrid = " $grid               			>> $namelist_file
echo " windf = '"$windf"'"           			>> $namelist_file
echo " im_global = "$im_global""                >> $namelist_file
echo " jm_global = "$jm_global""                >> $namelist_file
echo " kb= "$kb""							    >> $namelist_file
echo " im_local= "$im_local""				    >> $namelist_file
echo " jm_local= "$jm_local""				    >> $namelist_file
echo " im_global_coarse= "$im_global_coarse""   >> $namelist_file
echo " jm_global_coarse= "$jm_global_coarse""   >> $namelist_file
echo " im_local_coarse= "$im_local_coarse""     >> $namelist_file
echo " jm_local_coarse= "$jm_local_coarse""     >> $namelist_file
echo " x_division= "$x_division""				>> $namelist_file
echo " y_division= "$y_division""				>> $namelist_file
echo " n_proc= "$n_proc""						>> $namelist_file
echo "/"                             			>> $namelist_file


if [ $debug -eq 1 ]; then
	echo "& pom_nml "								>  $namelist_file_debug
	echo " title = '"$title"'"						>> $namelist_file_debug
	echo " netcdf_file = " $netcdf_file				>> $namelist_file_debug
	echo " mode = " $mode                			>> $namelist_file_debug
	echo " nadv = " $nadv                			>> $namelist_file_debug
	echo " nitera = " $nitera            			>> $namelist_file_debug
	echo " sw = " $sw                    			>> $namelist_file_debug
	echo " npg = " $npg                  			>> $namelist_file_debug
	echo " dte = " $dte                  			>> $namelist_file_debug
	echo " isplit = " $isplit            			>> $namelist_file_debug
	echo " time_start ='"$date_startfix"_00:00:00'" >> $namelist_file_debug
	echo " nread_rst = " $nread_rst					>> $namelist_file_debug
	echo " read_rst_file = '"$restart_file"'"		>> $namelist_file_debug
	echo " write_rst = " $write_rst					>> $namelist_file_debug
	echo " write_rst_file = " $write_rst_file		>> $namelist_file_debug
	echo " days = " $days							>> $namelist_file_debug
	echo " prtd1 = " $prtd1              			>> $namelist_file_debug
	echo " prtd2 = " $prtd2              			>> $namelist_file_debug
	echo " iperx = " $iperx              			>> $namelist_file_debug
	echo " ipery = " $ipery              			>> $namelist_file_debug
	echo " n1d = "   $n1d                			>> $namelist_file_debug
	echo " ngrid = " $grid               			>> $namelist_file_debug
	echo " windf = '"$windf"'"           			>> $namelist_file_debug
	echo "/"                             			>> $namelist_file_debug
fi

echo " "
echo "Run : from "$date_start" to " $date_end"."
echo " Creating namelist files .... "

#----------------------------------------------------------------
#                   creating switch.nml
# ---------------------------------------------------------------

switch_namelist="switch.nml."$date_start
echo '& switch_nml' > $switch_namelist

if [ $wind -eq 1 ]; then
	echo 'calc_wind=.true.'      >> $switch_namelist
else
	echo 'calc_wind=.false.'     >> $switch_namelist
fi

if [ $tsforce -eq 1 ]; then
	echo 'calc_tsforce=.true.'   >> $switch_namelist
else
	echo 'calc_tsforce=.false.'  >> $switch_namelist
fi

if [ $river -eq 1 ]; then
	echo 'calc_river=.true.'     >> $switch_namelist
else
	echo 'calc_river=.false.'    >> $switch_namelist
fi

if [ $assimssh -eq 1 ]; then
	echo 'calc_assim=.true.'     >> $switch_namelist
else
	echo 'calc_assim=.false.'    >> $switch_namelist
fi

if [ $assimdrf -eq 1 ]; then
	echo 'calc_assimdrf=.true.'  >> $switch_namelist
else
	echo 'calc_assimdrf=.false.' >> $switch_namelist
fi

if [ $assimsst -eq 1 ]; then
	echo 'calc_tsurf_mc=.true.'  >> $switch_namelist
else
	echo 'calc_tsurf_mc=.false.' >> $switch_namelist
fi

if [ $tide -eq 1 ]; then
	echo 'calc_tide=.true.'      >> $switch_namelist
else
	echo 'calc_tide=.false.'     >> $switch_namelist
fi

if [ $trajdrf -eq 1 ]; then
	echo 'calc_trajdrf=.true.'   >> $switch_namelist
else
	echo 'calc_trajdrf=.false.'  >> $switch_namelist
fi

echo 'tracer_flag=0' >> $switch_namelist

if [ $stokes -eq 1 ]; then
	echo 'calc_stokes=.true.'    >> $switch_namelist
else
	echo 'calc_stokes=.false.'   >> $switch_namelist
fi

if [ $vort -eq 1 ]; then
	echo 'calc_vort=.true.'      >> $switch_namelist
else
	echo 'calc_vort=.false.'     >> $switch_namelist
fi


#?output_flag: 0 -> separate *.nc files ; 1 -> one big file
echo 'output_flag=1'			 >> $switch_namelist 
#?SURF_flag: 0 -> no output SURF file; 1 -> output SRF.*.nc
echo 'SURF_flag=1'				 >> $switch_namelist 
#
echo '/'						 >> $switch_namelist
echo ' '						 >> $switch_namelist

echo '& assim_nml'				 >> $switch_namelist
echo 'nassim=1.'				 >> $switch_namelist
echo '/'						 >> $switch_namelist

if [ $assimdrf -eq 1 ]; then
	echo ' '					 >> $switch_namelist
	echo '& assimdrf_nml'		 >> $switch_namelist
	echo 'nassimdrf=0.125'		 >> $switch_namelist #?3hourly drifter assimilation
	echo 'nx=5.'				 >> $switch_namelist
	echo 'ny=5.'				 >> $switch_namelist
	echo 'beta=1.'				 >> $switch_namelist
	echo 'mindist=0.01'			 >> $switch_namelist
	echo 'zd=30.'				 >> $switch_namelist
	echo '/'					 >> $switch_namelist
fi

#
if [ $trajdrf -eq 1 ]; then
	echo ' '					 >> $switch_namelist
	echo '& trajdrf_nml'		 >> $switch_namelist
	echo 'ntrajdrf=0.04167'		 >> $switch_namelist #?1hourly drifer tracking
	echo '/'					 >> $switch_namelist
fi

rm -f pom.nml switch.nml
ln -s  pom.nml.$date_start pom.nml
if [ $debug -eq 1 ]; then
	ln -s  pom_debug.nml.$date_start pom_debug.nml
fi

ln -s  switch.nml.$date_start switch.nml

#
# ---------------------------------------------------------------
# ---------------------------------------------------------------
echo " Done. "
# ---------------------------------------------------------------
# ---------------------------------------------------------------
#
# -----------Creating batch job script file -----------------------
sh_filename=$title.$date_start.sh
log_filename=$title.$date_start.log
echo " Creating batch job script .... "$sh_filename 

echo "#\!/bin/sh  "              >  $sh_filename
echo "#BSUB -n $num_of_nodes "   >> $sh_filename
echo "#BSUB -q hpc_linux"		 >> $sh_filename
echo "#BSUB -k eo"				 >> $sh_filename
echo "#BSUB -J $title"			 >> $sh_filename
echo ' '						 >> $sh_filename
echo 'set time1=`date +%s`'		 >> $sh_filename 
echo ' '						 >> $sh_filename 
echo "#initialize environment modules"  >> $sh_filename
echo "  unset echo"				 >> $sh_filename
echo "  set echo "				 >> $sh_filename
echo "# ----------------"        >> $sh_filename
echo 'cd '$rundir				 >> $sh_filename
echo ' '						 >> $sh_filename 
echo 'rm -f pom.nml switch.nml'  >> $sh_filename
echo 'ln -s  pom.nml.'$date_start pom.nml						  >> $sh_filename
echo 'ln -s  switch.nml.'$date_start switch.nml				  	  >> $sh_filename

if [ $trajdrf -eq 1 ]; then
	echo 'ln -s '$homdir'/drf.list' drf.list				  	  >> $sh_filename
fi

if [ $nread_rst -eq 1 ]; then
	echo ' '					 >> $sh_filename 
	echo 'if [ -f  `echo out/'$restart_file'` ]; then'		  	  >> $sh_filename
	echo '    cd in '			 >> $sh_filename 
	echo '	  ln -s ../out/'$restart_file' '				  	  >> $sh_filename
	echo '	  cd .. '			 >> $sh_filename
	echo 'elif [ -f `echo in/'$restart_file'` ]; then'			  >> $sh_filename
	echo '    echo "Restart file: "'$restart_file 'found in in/'  >> $sh_filename
	echo 'else '			     >> $sh_filename
	echo '    echo "Restart file not found : " '$restart_file	  >> $sh_filename
	echo '    exit 1 '			 >> $sh_filename
	echo 'endif '				 >> $sh_filename
fi

echo ' '						 >> $sh_filename 
echo ' '						 >> $sh_filename 
echo 'export MPI_TYPE_DEPTH=10'  >> $sh_filename
echo 'export MPI_TYPE_MAX=1000000' >> $sh_filename
echo 'export F_UFMTENDIAN="big"' >> $sh_filename
echo ' '						 >> $sh_filename 

echo "mpirun -np $num_of_nodes ./pom.exe " >> $sh_filename
echo ' '								   >> $sh_filename 
echo 'time2=`date +%s`'					   >> $sh_filename 
echo 'dsec = $[$time2 - $time1]'		   >> $sh_filename 
echo 'dhour = $[$dsec/3600]'			   >> $sh_filename 
echo 'dsec = $[$dsec%3600]'				   >> $sh_filename 
echo 'dmin = $[$dsec/60]'				   >> $sh_filename 
echo 'dsec = $[$dsec%60]'				   >> $sh_filename 
echo 'echo "time : "$dhour"h "$dmin"m "$dsec"s "  '  >> $sh_filename 
echo ' '								   >> $sh_filename 

if [ "$date_end" != "$date_end0" ]; then
	sh_filename2=$title.$date_end.sh
	echo 'chmod 744 '$rundir'/'$csh_filename2 >> $sh_filename 
	echo 'module load intel_poe'  >>$sh_filename
	echo 'bsub '$rundir'/'$sh_filename2 >>$sh_filename
fi

echo ' '						  >> $sh_filename 
echo 'exit 0'					  >> $sh_filename 
# ---------------------------------------------------------------
echo " Done. "
date_start=$date_end
# ---------------------------------------------------------------
#end
# ---------------------------------------------------------------
# ---------------------------------------------------------------
echo " "
echo "bsub run : $title.$date_start0.sh"

cd $rundir
mpirun -n $num_of_nodes ./pom.exe

# ---------------------------------------------------------------
chmod 744 "$title.$date_start0.sh"
#bsub "$rundir/$title.$date_start0.csh"  &
# ---------------------------------------------------------------

echo $date_end0
echo $title
echo $rundir


