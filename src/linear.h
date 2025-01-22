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
    // Init to 1 here

    size_t shape_b[] = {10, 1};
    model->biases  = mdarray_create(2, shape_b, sizeof(double));
    // Init to 0 here

    return model;
}

MDArray* linearmodel_forward(LinearModel* model) {
    size_t idx[] = {0};
    MDArray* img = mdarray_copy(model->images, 1, idx);

    size_t shape_f[] = {784, 1};
    MDArray* img_flatten = mdarray_resize(img, 2, shape_f);

    return mdarray_dot(model->weights, img_flatten); // TODO: sum with biases
}

LinearModel* loss_function(LinearModel* model) {
// implement svm
    return model;
}
