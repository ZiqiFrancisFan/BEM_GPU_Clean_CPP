objects_main = main.o numerical.o mesh.o GMRES.o atomicFuncs.o

main : $(objects_main)
	nvcc -ccbin /usr/bin/gcc -arch=sm_60 -l=curand -l=cublas -lcusolver -L/usr/local/lib -lgsl -lgslcblas $(objects_main) -o main.out

main.o : main.cpp numerical.cu numerical.h mesh.cu mesh.h
	g++ -I/usr/local/cuda-9.1/include -std=c++11 -c main.cpp

numerical.o : numerical.cu numerical.h mesh.h
	nvcc -arch=sm_60 -dc -std=c++11 numerical.cu
	
mesh.o: mesh.cu mesh.h numerical.h
	nvcc -arch=sm_60 -dc -std=c++11 mesh.cu
	
GMRES.o: GMRES.cu GMRES.h numerical.h
	nvcc -arch=sm_60 -dc -std=c++11 GMRES.cu
	
atomicFuncs.o: atomicFuncs.cu atomicFuncs.h numerical.h
	nvcc -arch=sm_60 -dc -std=c++11 atomicFuncs.cu

clean :
	rm main.out $(objects_main)