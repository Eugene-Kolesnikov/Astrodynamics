NVCC        = nvcc
NVCCFLAGS = -arch=sm_30 -O3 -rdc=true \
 	-lglfw3 -Xlinker -framework,Cocoa -Xlinker -framework,OpenGL -Xlinker -framework,IOKit -Xlinker -framework,CoreVideo

OBJECTS = main.o \
	abstractsimulation.o \
	cpu_simulation.o \
	cpu_computations.o \
	gpu_simulation.o \
	gpu_computations.o \
	gpu_bh_simulation.o \
	gpu_bh_computations.o \
	floatComputations.o

all: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o astrodynamics

%.o: %.cc
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm *.o
