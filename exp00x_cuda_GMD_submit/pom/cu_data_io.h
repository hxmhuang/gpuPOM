#ifndef CU_DATA_IO_H
#define CU_DATA_IO_H


extern float *uab_mean_io;
extern float *vab_mean_io;
extern float *elb_mean_io;
extern float *wusurf_mean_io;
extern float *wvsurf_mean_io;
extern float *wtsurf_mean_io;
extern float *wssurf_mean_io;

extern float *uab_io;
extern float *vab_io;
extern float *elb_io;
extern float *et_io;
extern float *wusurf_io;
extern float *wvsurf_io;
extern float *wtsurf_io;
extern float *wssurf_io;

////////////////////////////////

extern float *u_mean_io;
extern float *v_mean_io;
extern float *w_mean_io;
extern float *t_mean_io;
extern float *s_mean_io;
extern float *rho_mean_io;
extern float *kh_mean_io;
extern float *km_mean_io;

extern float *u_io;
extern float *v_io;
extern float *w_io;
extern float *t_io;
extern float *s_io;
extern float *rho_io;
extern float *kh_io;
extern float *km_io;


#ifdef __NVCC__
extern "C" {
#endif

	void init_cuda_ipc_io();
	void data_copy2H_io(int num_out);

#ifdef __NVCC__
}
#endif

#endif
