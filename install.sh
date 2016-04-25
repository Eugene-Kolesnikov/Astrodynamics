nvcc -o run main.cu NBodySimulation.cu Registry.cpp -lglfw3 -Xlinker -framework,Cocoa -Xlinker -framework,OpenGL -Xlinker -framework,IOKit -Xlinker -framework,CoreVideo
