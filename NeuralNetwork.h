// NAME: V Sai Sandhya Kakarlamudi
// ID: 2020H1030122H
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdarg.h>

struct layer;

enum layer_type {
  INPUT_LAYER,
  HIDDEN_LAYER,
  OUTPUT_LAYER,
};

struct perceptron {
  double learning_rate;
  int number_of_inputs;
  double delta;
  double *weights;
  double previous_output;
  struct layer *layer;
};

struct layer {
  enum layer_type type;
  struct layer *next;
  struct layer *previous;
  int number_of_perceptrons;
  struct perceptron **perceptrons;
};

struct neural_network {
  int number_of_layers;
  struct layer **layers;
};

struct perceptron *perceptron_new(struct layer *layer, int number_of_inputs,
                                  double learning_rate);

void perceptron_delete(struct perceptron *perceptron);

struct layer *layer_new(enum layer_type type, int size, double learning_rate,
                        struct layer *prev, struct layer *next,
                        int perceptron_input_size);

void layer_delete(struct layer *layer);

double feed_perceptron(struct perceptron **perceptron, double *input_vec,
                       double bias);

void feed_layer(struct layer *layer, double *prev_layer_output,
                double *layer_output);

struct neural_network *neural_network_new(int number_of_layers,
                                          double learning_rate, ...);

void neural_network_delete(struct neural_network *ann);

double classify(struct neural_network *ann, double *input_vec,
              double *output_vec);

void back_propogate(struct neural_network *ann, double *expected_result_vec);

void train_one(struct neural_network *ann, double *input_vector,
               double *expected_result_vector);

int test_one(struct neural_network *ann,int number_of_rows,int feature_vector_size,
          int class_vector_size,double (*input_vector)[feature_vector_size],
          double (*test_values)[class_vector_size]);

void train(struct neural_network *ann, int number_of_rows,
           int feature_vector_size, int class_vector_size,
           double (*feature_rows)[feature_vector_size],
           double (*classification_rows)[class_vector_size],
           int number_of_epochs);

#endif
