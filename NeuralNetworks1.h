// NAME: V Sai Sandhya Kakarlamudi
// ID: 2020H1030122H
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <omp.h>
#include "NeuralNetwork.h"

#define GET(LAYER_PTR, INDEX) ((LAYER_PTR)->perceptrons[(INDEX)])
#define LAYER(ANN, LAYER_INDEX) ((ANN)->layers[(LAYER_INDEX)])

static double dot_product(int vec_size, double *first_vec, double *second_vec) {
	double dot_product = 0.0;

	for (int i = 0; i < vec_size; ++i){
		dot_product += first_vec[i] * second_vec[i];
	}

	return dot_product;
}

static void vector_add(int vec_size, double *result_vec, double *first_vec, double *second_vec) {
	for (int i = 0; i < vec_size; ++i) {
		result_vec[i] = first_vec[i] + second_vec[i];
	}
}

static double sigmoid(double x) {
	return (1.0 / (1 + exp(-x)));
}

static double sigmoid_derivative(double sigmoid_value) {
	return sigmoid_value * (1 - sigmoid_value);
}

double feed_perceptron(struct perceptron **perceptron, double *input_vec,
                       double bias) {

  switch ((*perceptron)->layer->type) {
  case INPUT_LAYER:
		(*perceptron)->previous_output = input_vec[(perceptron - &((*perceptron)->layer->perceptrons[0]))];
    break;
  case OUTPUT_LAYER:
  case HIDDEN_LAYER:
    (*perceptron)->previous_output =
        sigmoid(bias + dot_product((*perceptron)->number_of_inputs, input_vec,
                                   (*perceptron)->weights));
    break;
  }
  return (*perceptron)->previous_output;
}

void feed_layer(struct layer *layer, double *prev_layer_output, double *layer_output) {
	assert(prev_layer_output && "Previous layer output is missing");
	assert(layer_output	&& "Cannot write layer output");


		for (int i = 0; i < layer->number_of_perceptrons; ++i) {
		layer_output[i] = feed_perceptron(&layer->perceptrons[i], prev_layer_output, 1.0);

}
}

struct perceptron *perceptron_new(struct layer *layer, int number_of_inputs,
                                  double learning_rate) {
  struct perceptron *new_perceptron = malloc(sizeof(struct perceptron));
  if (new_perceptron) {
    new_perceptron->number_of_inputs = number_of_inputs;
    new_perceptron->learning_rate = learning_rate;
    new_perceptron->layer = layer;
    new_perceptron->previous_output = 0.0;
    new_perceptron->delta = 0.0;
    new_perceptron->weights = malloc(sizeof(double) * number_of_inputs);

    for (int i = 0; i < number_of_inputs; ++i)
      new_perceptron->weights[i] = (double)random() / RAND_MAX;
      }

  return new_perceptron;
}

void perceptron_delete(struct perceptron *perceptron) {
  free(perceptron->weights);
	free(perceptron);
}

struct layer *layer_new(enum layer_type type, int size, double learning_rate,
                        struct layer *prev, struct layer *next,
                        int perceptron_input_size) {
  struct layer *new_layer = malloc(sizeof(struct layer));
  if (new_layer) {
		//number of perceptrons = number of features
    new_layer->number_of_perceptrons = size;
    new_layer->type = type;
    new_layer->perceptrons = malloc(sizeof(struct perceptron) * size);

    for (int i = 0; i < size; ++i)
      new_layer->perceptrons[i] =
          perceptron_new(new_layer, perceptron_input_size, learning_rate);

    switch (new_layer->type) {
    case INPUT_LAYER:
      new_layer->previous = NULL;
      new_layer->next = next;
      if (next) {
        next->previous = new_layer;
        assert(next->perceptrons[0]->number_of_inputs ==
                   new_layer->number_of_perceptrons &&
               "Number of perceptrons in current layer should be equal to "
               "number of inputs in each perceptron in the next layer");
      }
      perceptron_input_size = 1;
      break;
    case HIDDEN_LAYER:
      new_layer->previous = prev;
      new_layer->next = next;
      if (prev) {
        prev->next = new_layer;
        assert(prev->number_of_perceptrons == new_layer->perceptrons[0]->number_of_inputs &&
               "Number of perceptrons in previous layer should be equal to "
               "number of inputs in each perceptron in the current layer");
      }
      if (next) {
        next->previous = new_layer;
        assert(next->perceptrons[0]->number_of_inputs ==
                   new_layer->number_of_perceptrons &&
               "Number of perceptrons in current layer should be equal to "
               "number of inputs in each perceptron in the next layer");
      }
      break;
    case OUTPUT_LAYER:
      new_layer->next = NULL;
      new_layer->previous = prev;
      if (prev) {
        prev->next = new_layer;
        assert(prev->number_of_perceptrons ==
                   new_layer->perceptrons[0]->number_of_inputs &&
               "Number of perceptrons in previous layer should be equal to "
               "number of inputs in each perceptron in the current layer");
      }
      break;
    default:
      assert(0 && "Unreachable!!");
    }
  }
  return new_layer;
}

void layer_delete(struct layer *layer) {
	if (layer) {
		for (int i = 0; i < layer->number_of_perceptrons; ++i)
			perceptron_delete(layer->perceptrons[i]);
	}
	free(layer->perceptrons);
	free(layer);
}

struct neural_network *
neural_network_new(int number_of_layers, double learning_rate, ...) {
  assert(number_of_layers >= 2 &&
         "There should be atleast one input layer, zero or more hidden layers, "
         "and one output layer !!");
  struct neural_network *ann = malloc(sizeof(struct neural_network) +
                                      sizeof(double *) * (number_of_layers));
  va_list arguments;
  int layer_size;

  if (ann) {
    ann->number_of_layers = number_of_layers;
    ann->layers = malloc(sizeof(struct layers *) * number_of_layers);
    va_start(arguments, learning_rate);
    layer_size = va_arg(arguments, int);
    ann->layers[0] =
        layer_new(INPUT_LAYER, layer_size, learning_rate, NULL, NULL, 1);
    va_end(arguments);
		layer_size = va_arg(arguments, int);

    for (int i = 1; i < (number_of_layers - 1); ++i) {
      //layer_size = va_arg(arguments, int);
      ann->layers[i] =
          layer_new(HIDDEN_LAYER, layer_size, learning_rate, ann->layers[i - 1],
                    NULL, ann->layers[i - 1]->number_of_perceptrons);
    }
    layer_size = va_arg(arguments, int);
    ann->layers[number_of_layers - 1] =
        layer_new(OUTPUT_LAYER, layer_size, learning_rate,
                  ann->layers[number_of_layers - 2], NULL,
                  ann->layers[number_of_layers - 2]->number_of_perceptrons);
  }
  return ann;
}

void neural_network_delete(struct neural_network *ann) {
	for (int i = 0; i < ann->number_of_layers; ++i)
		layer_delete(ann->layers[i]);
	free(ann->layers);
	free(ann);
}

double classify(struct neural_network *ann, double *input_vec,
              double *output_vec) {
  double *input =
      malloc(sizeof(double) * ann->layers[0]->number_of_perceptrons);
  memcpy(input, input_vec,
         sizeof(double) * ann->layers[0]->number_of_perceptrons);
  double *output;

  for (int i = 0; i < ann->number_of_layers; ++i) {
    output = malloc(sizeof(double) * ann->layers[i]->number_of_perceptrons);
    feed_layer(ann->layers[i], input, output);
    free(input);
    input = output;
  }
  if (output_vec)
    memcpy(output_vec, output,
           sizeof(double) *
               ann->layers[ann->number_of_layers - 1]->number_of_perceptrons);
 return *output;
}

void back_propogate(struct neural_network *ann, double *expected_result_vec) {
	struct layer *output_layer = LAYER(ann, ann->number_of_layers - 1);
	struct layer *hidden_layer;

  for (int i = 0; i < output_layer->number_of_perceptrons; ++i)
    GET(output_layer, i)->delta = sigmoid_derivative(GET(output_layer, i)->previous_output) *
              										(expected_result_vec[i] - GET(output_layer, i)->previous_output);

  for (int i = ann->number_of_layers - 2; i > 0; --i) {
		hidden_layer = LAYER(ann, i);
		for (int h = 0; h < hidden_layer->number_of_perceptrons; ++h) {
			double summation = 0.0;
			for (int k = 0; k < LAYER(ann, i+1)->number_of_perceptrons; ++k)
				summation += GET(LAYER(ann, i+1), k)->weights[h] * GET(LAYER(ann, i+1), k)->delta;
			GET(hidden_layer, h)->delta = sigmoid_derivative(GET(hidden_layer, h)->previous_output) * summation;
		}
  }
  for (int i = ann->number_of_layers - 1; i > 0; --i) {
		for (int h = 0; h < LAYER(ann, i)->number_of_perceptrons; ++h) {
			for (int idx = 0; idx < GET(LAYER(ann, i), h)->number_of_inputs; ++idx) {
				GET(LAYER(ann, i), h)->weights[idx] += GET(LAYER(ann, i), h)->learning_rate
																						 * GET(LAYER(ann, i), h)->delta
																						 * GET(LAYER(ann, i - 1), idx)->previous_output;
			}
		}
	}
}

void train_one(struct neural_network *ann, double *input_vector,
               double *expected_result_vector) {
	/*for(int i=0;i<sizeof(input_vector);i++){
		printf("%lf\t",input_vector[i]);
	}*/
  double dup = classify(ann, input_vector, NULL);
	//printf("%lf\t",dup);
  back_propogate(ann, expected_result_vector);
}

int test_one(struct neural_network *ann,int number_of_rows,int feature_vector_size,
	 int class_vector_size,double (*input_vector)[feature_vector_size],
	 double (*test_values)[class_vector_size]) {
		int c=0;

	for(int i=0;i<number_of_rows;i++){
			double out = classify(ann, input_vector[i], NULL);
			//printf("%lf\t%lf\n",out,test_values[i][0]);
			if(test_values[i][0]-out <= 0.01){
				c++;
			}

	}
	return c;
}

void train(struct neural_network *ann, int number_of_rows, int feature_vector_size,
           int class_vector_size, double (*feature_rows)[feature_vector_size],
           double (*classification_rows)[class_vector_size], int number_of_epochs) {
           #pragma omp parallel
	{
		int threadid=omp_get_thread_num();
		#pragma omp parallel for schedule(static,256)
	for (int i = 0; i < number_of_epochs; ++i) {
		for (int j = 0; j < number_of_rows; ++j) {
			train_one(ann, feature_rows[j], classification_rows[j]);
		}
		}
	}
}
