objects_main = main.o numerical.o

main : $(objects_main)
	nvcc -ccbin /usr/bin/gcc -arch=sm_50 -l=curand -l=cublas -L/usr/local/lib -lgsl -lgslcblas $(objects_main) -o main.out

main.o : main.cpp numerical.cu numerical.h
	g++ -I/usr/local/cuda-9.1/include -std=c++11 -c main.cpp

numerical.o : numerical.cu numerical.h
	nvcc -arch=sm_50 -dc -std=c++11 numerical.cu

clean :
	rm main.out $(objects_main)