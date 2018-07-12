#Author - Anuraag Motiwale asmotiwa@ncsu.edu
#Author - Abhishek Singh aksingh5@ncsu.edu


#This is the test script to test the tool generated in clang

#move the source code for all the transformations to respective folders in clang

sudo mv transformation1/ /home/ubuntu/llvm/llvm/tools/clang/tools/extra

sudo mv transformation2/ /home/ubuntu/llvm/llvm/tools/clang/tools/extra

sudo mv transformation3/ /home/ubuntu/llvm/llvm/tools/clang/tools/extra

sudo mv transformation4/ /home/ubuntu/llvm/llvm/tools/clang/tools/extra

sudo echo 'add_subdirectory(transformation1)' >> /home/ubuntu/llvm/llvm/tools/clang/tools/extra/CMakeLists.txt

sudo echo 'add_subdirectory(transformation2)' >> /home/ubuntu/llvm/llvm/tools/clang/tools/extra/CMakeLists.txt

sudo echo 'add_subdirectory(transformation3)' >> /home/ubuntu/llvm/llvm/tools/clang/tools/extra/CMakeLists.txt

sudo echo 'add_subdirectory(transformation4)' >> /home/ubuntu/llvm/llvm/tools/clang/tools/extra/CMakeLists.txt

## Build the tools
cd ../llvm/build-release/
sudo ninja

## Run the transformations

/home/ubuntu/llvm/build-release/bin/transformation1 /home/ubuntu/CC_Project/Cuda_code/mst_dp_modular/main.cu -- --cuda-host-only --cuda-gpu-arch=sm_35 -L /usr/local/cuda/include -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/include/ -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/cub-1.7.4/ -I /usr/local/cuda/pthread > /home/ubuntu/CC_Project/Cuda_code/mst_dp_T1_modular/main.cu -w

/home/ubuntu/llvm/build-release/bin/transformation2 /home/ubuntu/CC_Project/Cuda_code/mst_dp_modular/main.cu -- --cuda-host-only --cuda-gpu-arch=sm_35 -L /usr/local/cuda/include -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/include/ -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/cub-1.7.4/ -I /usr/local/cuda/pthread > /home/ubuntu/CC_Project/Cuda_code/mst_dp_T2_modular/main.cu -w

/home/ubuntu/llvm/build-release/bin/transformation3 /home/ubuntu/CC_Project/Cuda_code/mst_dp_modular/main.cu -- --cuda-host-only --cuda-gpu-arch=sm_35 -L /usr/local/cuda/include -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/include/ -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/cub-1.7.4/ -I /usr/local/cuda/pthread > /home/ubuntu/CC_Project/Cuda_code/mst_dp_T3_modular/main.cu -w

/home/ubuntu/llvm/build-release/bin/transformation4 /home/ubuntu/CC_Project/Cuda_code/mst_dp_modular/main.cu -- --cuda-host-only --cuda-gpu-arch=sm_35 -L /usr/local/cuda/include -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/include/ -I /home/ubuntu/CC_Project/Cuda_code/lonestargpu-2.0/cub-1.7.4/ -I /usr/local/cuda/pthread > /home/ubuntu/CC_Project/Cuda_code/mst_dp_T4_modular/main.cu -w



