#ifndef CSOLVER_H
#define CSOLVER_H

#include"data.h"

void dens(float si[][j_size][i_size], 
		  float ti[][j_size][i_size],
		  float rhoo[][j_size][i_size]);

void baropg_mcc();
/*
void baropg_mcc(float rho[][j_size][i_size], 
				float rmean[][j_size][i_size], 
				float d[][i_size],
				float dum[][i_size], 
				float dvm[][i_size],
				float dt[][i_size],
				float drhox[][j_size][i_size],
				float drhoy[][j_size][i_size],
				float ramp);
*/

void baropg();
/*
void baropg(float rho[][j_size][i_size], 
			float rmean[][j_size][i_size], 
		    float dum[][i_size], 
		    float dvm[][i_size],
			float dt[][i_size], 
			float drhox[][j_size][i_size],
			float drhoy[][j_size][i_size], 
			float ramp);
*/

void advct();
/*
void advct_(float advx[][j_size][i_size], float v[][j_size][i_size],
		   float u[][j_size][i_size], float dt[][i_size], 
		   float ub[][j_size][i_size], float aam[][j_size][i_size],
		   float vb[][j_size][i_size], float advy[][j_size][i_size]);
*/

void stokes();

/*
void advave(float advua[][i_size], float d[][i_size],
		     float ua[][i_size], float va[][i_size],
			 float uab[][i_size], float aam2d[][i_size],
			 float vab[][i_size], float advva[][i_size],
			 float wubot[][i_size], float wvbot[][i_size]);
*/

void advave();

void vort_curl(float fx[][i_size], float fy[][i_size],
			   float dx[][i_size], float dy[][i_size],
			   float dum[][i_size], float dvm[][i_size],
			   int im, int jm,
			   float cf[][i_size]);

void vort();

void realvertvl();

void vertvl();

void advq(float qb[][j_size][i_size], 
		  float q[][j_size][i_size],
		  float qf[][j_size][i_size]);

void profq();

void advt1(float fb[][j_size][i_size], 
		   float f[][j_size][i_size],
		   float fclim[][j_size][i_size], 
		   float ff[][j_size][i_size],
		   char var);

void advt2(float fb[][j_size][i_size],
		   float f[][j_size][i_size],
		   float fclim[][j_size][i_size],
		   float ff[][j_size][i_size],
		   char var);

void smol_adif(float xmassflux[][j_size][i_size], 
			   float ymassflux[][j_size][i_size], 
			   float zwflux[][j_size][i_size],
			   float ff[][j_size][i_size]);

void proft(float f[][j_size][i_size], 
		   float wfsurf[][i_size],
		   float fsurf[][i_size], 
		   int nbc);

void advu();

void advv();

void profu();

void profv();

#endif
