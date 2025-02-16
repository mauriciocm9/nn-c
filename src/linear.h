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
    mdarray_ones(model->weights);

    size_t shape_b[] = {10, 1};
    model->biases  = mdarray_create(2, shape_b, sizeof(double));
    mdarray_ones(model->biases);

    return model;
}

MDArray* linearmodel_forward(LinearModel* model) {
    size_t idx[] = {0};
    MDArray* img = mdarray_copy(model->images, 1, idx);

    size_t shape_f[] = {784, 1};
    MDArray* img_flatten = mdarray_resize(img, 2, shape_f);
    /*
    size_t shape2[] = {152, 0};
    double x = *(double*)mdarray_get_element(img_flatten, shape2);
    printf("index %f\n", x);
    for(size_t i = 0; i < 784; i++) {
    for(size_t j = 0; j < 1; j++) {
            size_t shape[] = {i, j}; // 5, 12 first non empty value
            double x = *(double*)mdarray_get_element(img_flatten, shape);
            if(x > 0.0) printf("someval %f\n", x);
        }
    }*/


    return mdarray_sum(mdarray_dot(model->weights, img_flatten), model->biases);
}

LinearModel* loss_function(LinearModel* model) {
    // implement svm
    return model;
}
