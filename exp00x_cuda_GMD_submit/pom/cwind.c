#include"cwind.h"
#include"data.h"

void wind_init(){
	int i, j;


	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			wusurf[j][i] = 0;	
			wvsurf[j][i] = 0;	
		}
	}

	if (ngrid == -1){
		for (j = 0; j < jm; j++){
			for (i = 0; i < im; i++){
				wusurf[j][i] = 0.7e-4f;;	
			}
		}
	}

	if (calc_wind){
		printf("calc_wind is true, but related functions"
			   "haven't been implemented now\n");	
		exit(1);
	}

	/*
	for (j = 0; j < jm; j++){
		for (i = 0; i < im; i++){
			f_wusurf[j][i] = wusurf[j][i];	
			f_wvsurf[j][i] = wvsurf[j][i];	
		}
	}
	*/
}

void wind_main(){

}
