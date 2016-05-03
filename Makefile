NVCC        = nvcc
NVCCFLAGS = -arch=sm_30 -O3 \
 	-lglfw3 -Xlinker -framework,Cocoa -Xlinker -framework,OpenGL -Xlinker -framework,IOKit -Xlinker -framework,CoreVideo

OBJECTS = NBodySimulation.o \
	Registry.o \
	computations.o \
	main.o

all: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o astrodynamics

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm *.o
