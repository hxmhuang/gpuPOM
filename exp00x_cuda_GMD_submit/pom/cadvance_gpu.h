#ifdef __NVCC__
extern "C"{
#endif
	void surface_forcing_gpu();
	void momentum3d_gpu();
	void mode_interaction_gpu();
	void mode_external_gpu();
	void mode_internal_gpu();
	void print_section_gpu();
	void check_velocity_gpu();
	void get_time_gpu();
	void store_mean_gpu();
	void store_surf_mean_gpu();
	void output_copy_back(int num);
	void advance_gpu();
	void advance_main_gpu();

#ifdef __NVCC__
}
#endif
