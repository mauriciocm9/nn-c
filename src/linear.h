#include <stdio.h>


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

    size_t shape_b[] = {10, 1};
    model->biases  = mdarray_create(2, shape_b, sizeof(double));

    return model;
}

MDArray* linearmodel_forward(LinearModel* model) {
    //return model->images * model->weights + model->biases;
    //we need images to be an array of nx768 probably, in theory is just plain we need
    //to alter just the strides, weigths is already in 10x768
    //also implement the multiply matrix
    
    // n, 28, 28
    // need to get a pointer to image for now of size 768x1

    // We just want one the first image for now
    size_t shape[] = {0};
    MDArray* img = mdarray_copy(model->images, 1, shape); // this is array 28x28
    size_t shape_f[] = {784, 1};
    MDArray* img_flatten = mdarray_resize(img, 2, shape_f); // this is array 28x28
    //printf("%zu %zu\n", model->images->ndim, model->images->shape[0]);
    printf("%zu %zu\n", img->ndim, img->shape[0]);

    return mdarray_dot(model->weights, img);
}

LinearModel* loss_function(LinearModel* model) {
// implement svm
    return model;
}
