objects_main = main.o numerical.o mesh.o GMRES.o

main : $(objects_main)
	nvcc -ccbin /usr/bin/gcc -arch=sm_50 -l=curand -l=cublas -L/usr/local/lib -lgsl -lgslcblas $(objects_main) -o main.out

main.o : main.cpp numerical.cu numerical.h mesh.cu mesh.h
	g++ -I/usr/local/cuda-9.1/include -std=c++11 -c main.cpp

numerical.o : numerical.cu numerical.h mesh.h
	nvcc -arch=sm_50 -dc -std=c++11 numerical.cu
	
mesh.o: mesh.cu mesh.h numerical.h
	nvcc -arch=sm_50 -dc -std=c++11 mesh.cu
	
GMRES.o: GMRES.cu GMRES.h
	nvcc -arch=sm_50 -dc -std=c++11 GMRES.cu

clean :
	rm main.out $(objects_main)