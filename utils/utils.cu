#include <iostream>
#include <thrust/host_vector.h>
#include <random>
#include <cmath> // For std::fabs

bool areFloatsEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}


int areSameMatrices(float* A_1, float* A_2, int M, int N){
    for (int i = 0; i < M * N; i++){
        if (!(areFloatsEqual(A_1[i], A_2[i]))) {
            //std::cout << "Wrong answer:" << A_1[i] << " " << A_2[i] << std::endl;
            return 0;
        }
    }
    return 1;
}


thrust::host_vector<float> generateRandomMatrix(int M, int N) {
    thrust::host_vector<float> A(M * N);

    // Create random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution range
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate random matrix
    for (int k = 0; k < M * N; k++) {
        float randomFloat = dis(gen);
        A[k] = randomFloat;
    }

    return A;
}