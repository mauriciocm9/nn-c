#include <stdio.h>
#include <stdlib.h>
#include "mdarray.h"

// Structure to hold array metadata
typedef struct {
    MDArray* images;
    MDArray* labels;

    MDArray* weights;
    MDArray* biases;
} LinearModel;


LinearModel* linearmodel_new(MDArray* images, MDArray* labels) {
    LinearModel* model = (LinearModel*)malloc(sizeof(LinearModel));
    if(!model) return NULL;

    model->images = images;
    model->labels = labels;

    size_t shape_w[] = {10, 28*28};
    model->weights = mdarray_create(2, shape_w, sizeof(double));
    mdarray_randn(model->weights, 0.01);

    size_t shape_b[] = {10, 1};
    model->biases  = mdarray_create(2, shape_b, sizeof(double));
    mdarray_zeros(model->biases);

    return model;
}

MDArray* linearmodel_forward(LinearModel* model) {
    size_t n = model->images->shape[0];

    // (N, 28, 28) -> (N, 784)
    size_t shape_flat[] = {n, 784};
    MDArray* imgs_flat = mdarray_resize(model->images, 2, shape_flat);

    // (N, 784) -> (784, N)
    MDArray* imgs_t = mdarray_transpose_2d(imgs_flat);
    mdarray_free(imgs_flat);

    // W(10, 784) * X(784, N) = (10, N)
    MDArray* scores = mdarray_dot(model->weights, imgs_t);
    mdarray_free(imgs_t);

    // Add biases (10, 1) broadcast to each column
    for (size_t i = 0; i < 10; i++) {
        size_t bias_idx[] = {i, 0};
        double b = *(double*)mdarray_get_element(model->biases, bias_idx);
        for (size_t j = 0; j < n; j++) {
            size_t idx[] = {i, j};
            double* s = (double*)mdarray_get_element(scores, idx);
            double val = *s + b;
            mdarray_set_element(scores, idx, &val);
        }
    }

    return scores;
}

MDArray* svm_loss_backward(MDArray* scores, size_t* labels, size_t batch_size) {
    size_t num_classes = scores->shape[0];
    size_t shape[] = {num_classes, batch_size};
    MDArray* dscores = mdarray_create(2, shape, sizeof(double));
    mdarray_zeros(dscores);

    for (size_t i = 0; i < batch_size; i++) {
        size_t yi = labels[i];
        size_t idx_correct[] = {yi, i};
        double s_yi = *(double*)mdarray_get_element(scores, idx_correct);

        size_t count = 0;
        for (size_t j = 0; j < num_classes; j++) {
            if (j == yi) continue;
            size_t idx_j[] = {j, i};
            double s_j = *(double*)mdarray_get_element(scores, idx_j);
            double margin = s_j - s_yi + 1.0;
            if (margin > 0.0) {
                double grad = 1.0 / batch_size;
                mdarray_set_element(dscores, idx_j, &grad);
                count++;
            }
        }

        double neg_grad = -(double)count / batch_size;
        mdarray_set_element(dscores, idx_correct, &neg_grad);
    }

    return dscores;
}

void linearmodel_backward(LinearModel* model, MDArray* scores, size_t* labels, size_t batch_size, double lr) {
    MDArray* dscores = svm_loss_backward(scores, labels, batch_size);

    // Reconstruct X_t (784, N) from images, same as forward pass
    size_t n = model->images->shape[0];
    size_t shape_flat[] = {n, 784};
    MDArray* imgs_flat = mdarray_resize(model->images, 2, shape_flat);
    MDArray* X_t = mdarray_transpose_2d(imgs_flat);
    mdarray_free(imgs_flat);

    // dW(10,784) = dscores(10,N) * X_t^T(N,784)
    MDArray* X_t_T = mdarray_transpose_2d(X_t);  // (N, 784)... wait, X_t is (784,N), transpose is (N,784)
    MDArray* dW = mdarray_dot(dscores, X_t_T);
    mdarray_free(X_t);
    mdarray_free(X_t_T);

    // db(10,1) = sum of dscores over columns
    for (size_t i = 0; i < dscores->shape[0]; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < batch_size; j++) {
            size_t idx[] = {i, j};
            sum += *(double*)mdarray_get_element(dscores, idx);
        }
        size_t bias_idx[] = {i, 0};
        double b = *(double*)mdarray_get_element(model->biases, bias_idx);
        double new_b = b - lr * sum;
        mdarray_set_element(model->biases, bias_idx, &new_b);
    }

    // Update weights: W -= lr * dW
    for (size_t i = 0; i < model->weights->shape[0]; i++) {
        for (size_t j = 0; j < model->weights->shape[1]; j++) {
            size_t idx[] = {i, j};
            double w = *(double*)mdarray_get_element(model->weights, idx);
            double dw = *(double*)mdarray_get_element(dW, idx);
            double new_w = w - lr * dw;
            mdarray_set_element(model->weights, idx, &new_w);
        }
    }

    mdarray_free(dscores);
    mdarray_free(dW);
}

double svm_loss(MDArray* scores, size_t* labels, size_t batch_size) {
    double total_loss = 0.0;
    size_t num_classes = scores->shape[0];

    for (size_t i = 0; i < batch_size; i++) {
        size_t yi = labels[i];
        size_t idx_correct[] = {yi, i};
        double s_yi = *(double*)mdarray_get_element(scores, idx_correct);

        for (size_t j = 0; j < num_classes; j++) {
            if (j == yi) continue;
            size_t idx_j[] = {j, i};
            double s_j = *(double*)mdarray_get_element(scores, idx_j);
            double margin = s_j - s_yi + 1.0;
            if (margin > 0.0) total_loss += margin;
        }
    }

    return total_loss / batch_size;
}
