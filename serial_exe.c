
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include "NeuralNetwork.h"
#include "NeuralNetworks.h"
#define NUMBER_OF_EPOCHS 1000
#define NUMBER_OF_FEATURES 17
#define NUMBER_OF_OUTPUTS 1

#define INPUT_LAYER_SIZE  NUMBER_OF_FEATURES
#define HIDDEN_LAYER_SIZE 10
#define OUTPUT_LAYER_SIZE NUMBER_OF_OUTPUTS
#define NUMBER_OF_LAYERS 5 

int main(int argc, char *argv[]) {
  clock_t start, end;
   double cpu_time_used;
   double dum;
   start = clock();
  srand(getpid());
  if (argc < 5) {
    fprintf(stderr, "Usage: ./a.out <train_file> <number_of_rows> <test_file> "
                    "<number_of_rows>");
    exit(1);
  }
  FILE *train_file = fopen(argv[1], "r");
    FILE *test_file  = fopen(argv[3], "r");
    int train_file_rows_n = atoi(argv[2]);
    int test_file_rows_n  = atoi(argv[4]);
    struct timespec start_time, end_time;

  double(*train_features)[NUMBER_OF_FEATURES] =
      malloc(sizeof(double[NUMBER_OF_FEATURES]) * train_file_rows_n);
  double(*train_outputs)[NUMBER_OF_OUTPUTS] =
      malloc(sizeof(double[NUMBER_OF_OUTPUTS]) * train_file_rows_n);
  double(*test_features)[NUMBER_OF_FEATURES] =
      malloc(sizeof(double[NUMBER_OF_FEATURES]) * test_file_rows_n);
  double(*test_outputs)[NUMBER_OF_OUTPUTS] =
      malloc(sizeof(double[NUMBER_OF_OUTPUTS]) * test_file_rows_n);
  double(*res)[NUMBER_OF_OUTPUTS] = NULL;
  assert(train_file && "Could not open training set file!!");
  assert(test_file  && "Could not open testing set file!!");

  timespec_get(&start_time, CLOCK_MONOTONIC);
  for (int i = 0; i < train_file_rows_n; ++i){
    fscanf(train_file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
           &train_features[i][0], &train_features[i][1],
           &train_features[i][2], &train_features[i][3], &train_features[i][4],
           &train_features[i][5], &train_features[i][6], &train_features[i][7],
           &train_features[i][8], &train_features[i][9], &train_features[i][10],
           &train_features[i][11],&train_features[i][12],&train_features[i][13],
           &train_features[i][14],&train_features[i][15],&train_features[i][16],
          &train_outputs[i][0]);

            //printf("%lf\t",train_features[i][1]);
}
  for (int i = 0; i < test_file_rows_n; ++i){
      fscanf(test_file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                     &test_features[i][0], &test_features[i][1],
                     &test_features[i][2], &test_features[i][3], &test_features[i][4],
                     &test_features[i][5], &test_features[i][6], &test_features[i][7],
                     &test_features[i][8], &test_features[i][9],&test_features[i][10],
                     &test_features[i][11],&test_features[i][12],&test_features[i][13],
                     &test_features[i][14],&test_features[i][15],&test_features[i][16],
                    &test_outputs[i][0]);
                    //printf("%lf\n",test_outputs[i][0]);
                  }
  struct neural_network *ann =
      neural_network_new(NUMBER_OF_LAYERS, 0.1, INPUT_LAYER_SIZE,
                         HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
  int correct = 0;
  train(ann, train_file_rows_n, NUMBER_OF_FEATURES, NUMBER_OF_OUTPUTS, train_features, train_outputs, NUMBER_OF_EPOCHS);


  correct = test_one(ann,test_file_rows_n, NUMBER_OF_FEATURES, NUMBER_OF_OUTPUTS,test_features,test_outputs);
  /*
   * Write code to find accuracy
   * Use test_features and test_outputs
   * Accuracy: (no. correct) / (test_file_rows_n)
   * Classification could be treated as correct
   * if | expected_output - obtained_output | <= error
   * the error value can be suitably chosen, say 0.01
   */

  timespec_get(&end_time, CLOCK_MONOTONIC);
  end = clock();
  free(train_features);
  free(train_outputs);
  free(test_features);
  free(test_outputs);
  printf("Accuracy: %.8lf", (double)(correct) / (test_file_rows_n));
  /// 1000000000.0
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("\nExecution Time: %f seconds", cpu_time_used);
  //printf("\nExecution Time: %ld seconds", (end_time.tv_sec + end_time.tv_nsec - start_time.tv_sec - start_time.tv_nsec) );
  neural_network_delete(ann);
  return 0;
}
