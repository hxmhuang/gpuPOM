
#ifdef __NVCC__
extern "C"{
#endif

	void bcond_gpu(int idx);
	void bcond_overlap(int idx, cudaStream_t &stream_in);

#ifdef __NVCC__
}
#endif
