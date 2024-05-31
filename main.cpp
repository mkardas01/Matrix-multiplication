#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <cstdlib>
#include "CL/cl.hpp"

using namespace std;
using namespace std::chrono;

void generateRandomMatrix(int rows, int cols, int** matrix, int minValue, int maxValue) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % (maxValue - minValue + 1) + minValue;
        }
    }
}

void mulMatSeq(int rows1, int cols1, int** mat1, int rows2, int cols2, int** mat2, int** rslt) {
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            rslt[i][j] = 0;

            for (int k = 0; k < cols1; k++) {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

void mulMatMP(int rows1, int cols1, int** mat1, int rows2, int cols2, int** mat2, int** rslt) {
#pragma omp parallel for private(i, j, k) shared(rows1, cols1, rows2, cols2, mat1, mat2, rslt)
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            rslt[i][j] = 0;
 
            for (int k = 0; k < cols1; k++) {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

void mulMatCL(int rows1, int cols1, int* mat1, int rows2, int cols2, int* mat2, int* rslt, vector<cl::Device> devices,  cl::CommandQueue& queue, cl::Context& context) {
    cl::Buffer buffer_mat1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * rows1 * cols1, mat1);
    cl::Buffer buffer_mat2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * rows2 * cols2, mat2);
    cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, sizeof(int) * rows1 * cols1);

            // Tworzymy kernel
    cl::Program::Sources sources;
    string kernel_code = R"(
                __kernel void matrixMul(__global const int* mat1, __global const int* mat2, __global int* result, const int rows1, const int cols1, const int cols2) {
                    int i = get_global_id(0);
                    int j = get_global_id(1);

                    int sum = 0;
                    for (int k = 0; k < cols1; ++k) {
                        sum += mat1[i * cols1 + k] * mat2[k * cols2 + j];
                    }
                    result[i * cols2 + j] = sum;
                }
            )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    cl::Program program(context, sources);
    program.build(devices);

    cl::Kernel kernel(program, "matrixMul");
    kernel.setArg(0, buffer_mat1);
    kernel.setArg(1, buffer_mat2);
    kernel.setArg(2, buffer_result);
    kernel.setArg(3, rows1);
    kernel.setArg(4, cols1);
    kernel.setArg(5, cols2);
	     

            // Wykonujemy kernel
    cl::NDRange global(rows1, cols2);
    cl::NDRange local(1, 1);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

            // Kopiujemy wynik z bufora
    queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, sizeof(int) * rows1 * cols2, rslt);

    

}

int main() {

    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " MIN_VALUE MAX_VALUE START_SIZE END_SIZE" << endl;
        return 1;
    }

    int minValue = atoi(argv[1]);
    int maxValue = atoi(argv[2]);
    int startSize = atoi(argv[3]);
    int endSize = atoi(argv[4]);

    if (minValue > maxValue){
        cerr << "Usage: " << argv[0] << " MIN_VALUE HAS TO BE SMALLER THAN MAX_VALUE" << endl;
        return 1;
    }

    if (startSize > endSize){
        cerr << "Usage: " << argv[0] << " START_SIZE HAS TO BE SMALLER THAN END_SIZE" << endl;
        return 1;
    }

    ofstream outputFile("results.txt");
    if (!outputFile.is_open()) {
        cerr << "Nie mo�na otworzy� pliku do zapisu!";
        return 1;
    }

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);

    outputFile << "Size; Seq; OpenMp; OpenCL;" << endl;

    for (int size = startSize; size <= endSize; size += 10) {
        int R1 = size;
        int C1 = size;
        int R2 = size;
        int C2 = size;

        cout << size << " ";

        int* mat1CL = new int[R1 * C1];
        int* mat2CL = new int[R2 * C2];
        int* rsltCL = new int[R1 * C2];

        int** mat1 = new int* [R1];
        for (int i = 0; i < R1; ++i) {
            mat1[i] = new int[C1];
        }

        int** mat2 = new int* [R2];
        for (int i = 0; i < R2; ++i) {
            mat2[i] = new int[C2];
        }

        int** rslt = new int* [R1];
        for (int i = 0; i < R1; ++i) {
            rslt[i] = new int[C2];
        }

        generateRandomMatrix(R1, C1, mat1);
        generateRandomMatrix(R2, C2, mat2);

        for (int i = 0; i < R1; ++i) {
            for (int j = 0; j < C1; ++j) {
                mat1CL[i * C1 + j] = mat1[i][j];
            }
        }

        for (int i = 0; i < R2; ++i) {
            for (int j = 0; j < C2; ++j) {
                mat2CL[i * C2 + j] = mat2[i][j];
            }
        }


        outputFile << size << ";";

        auto start_seq = high_resolution_clock::now();
        mulMatSeq(R1, C1, mat1, R2, C2, mat2, rslt); // Sequential
        auto stop_seq = high_resolution_clock::now();
        auto duration_seq = duration_cast<milliseconds>(stop_seq - start_seq);
        outputFile << duration_seq.count() << ";";
        std::cout << "Seq " << duration_seq.count() <<  " ";


        auto start_mp = high_resolution_clock::now();
        mulMatMP(R1, C1, mat1, R2, C2, mat2, rslt); // OpenMP
        auto stop_mp = high_resolution_clock::now();
        auto duration_mp = duration_cast<milliseconds>(stop_mp - start_mp);
        outputFile << duration_mp.count() << ";";
        std::cout << "OpenMP " << duration_mp.count() <<  " ";


        auto start_cl = high_resolution_clock::now();
        mulMatCL(R1, C1, mat1CL, R2, C2, mat2CL, rsltCL, devices, queue, context); // OpenCL
        auto stop_cl = high_resolution_clock::now();
        auto duration_cl = duration_cast<milliseconds>(stop_cl - start_cl);
        outputFile << duration_cl.count() << ";" << endl;
        std::cout << "OpenCL " << duration_cl.count() << endl;


        // Free dynamically allocated memory
        for (int i = 0; i < R1; ++i) {
            delete[] mat1[i];
        }
        delete[] mat1;

        for (int i = 0; i < R2; ++i) {
            delete[] mat2[i];
        }
        delete[] mat2;

        for (int i = 0; i < R1; ++i) {
            delete[] rslt[i];
        }
        delete[] rslt;

        delete[] mat1CL;
        delete[] mat2CL;
        delete[] rsltCL;

    }

    outputFile.close(); // Close the file

    return 0;
}
