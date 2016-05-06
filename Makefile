NVCC        = nvcc
NVCCFLAGS = -arch=sm_30 -O3 \
 	-lglfw3 -Xlinker -framework,Cocoa -Xlinker -framework,OpenGL -Xlinker -framework,IOKit -Xlinker -framework,CoreVideo

OBJECTS = main.o \
	abstractsimulation.o \
	cpu_simulation.o \
	gpu_simulation.o \
	computations.o

all: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o astrodynamics

%.o: %.cc
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm *.o
