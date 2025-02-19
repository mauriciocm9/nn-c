#include "mdarray.h"
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

MDArray* mdarray_create(size_t ndim, size_t* shape, size_t itemsize) {
    MDArray* arr = (MDArray*)malloc(sizeof(MDArray));
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->itemsize = itemsize;

    // Allocate and copy shape array
    arr->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(arr->shape, shape, ndim * sizeof(size_t));

    // Calculate strides
    arr->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->strides) {
        free(arr->shape);
        free(arr);
        return NULL;
    }

    arr->total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->total_size *= shape[i];
    }

    size_t stride = 1;
    for (size_t i = ndim - 1; i < ndim; i--) {
        arr->strides[i] = stride;
        stride *= shape[i];
    }

    // Allocate data array
    arr->data = malloc(arr->total_size * itemsize);
    if (!arr->data) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }

    return arr;
}


void mdarray_free(MDArray* arr) {
    if (arr) {
        free(arr->data);
        free(arr->shape);
        free(arr->strides);
        free(arr);
    }
}

size_t mdarray_calculate_index(MDArray* arr, size_t* indices) {
    size_t flat_index = 0;
    for (size_t i = 0; i < arr->ndim; i++) {
        if (indices[i] >= arr->shape[i]) {  // Add bounds checking
            printf("Index out of bounds: indices[%zu]=%zu >= shape[%zu]=%zu\n", i, indices[i], i, arr->shape[i]);
            return (size_t)-1;  // Return max value to indicate error
        }
        flat_index += indices[i] * arr->strides[i];
    }
    return flat_index;
}

void* mdarray_get_element(MDArray* arr, size_t* indices) {
    size_t index = mdarray_calculate_index(arr, indices);
    if (index == (size_t)-1) {  // Check for the error value
        return NULL;
    }
    return (char*)arr->data + (index * arr->itemsize);
}

void mdarray_set_element(MDArray* arr, size_t* indices, void* value) {
    size_t index = mdarray_calculate_index(arr, indices);
    memcpy((char*)arr->data + (index * arr->itemsize), value, arr->itemsize);
}

// mdarray_dot this is like matmul. Example:
// x       10x768
// y       768x1
// RETURNS 10x1
MDArray* mdarray_dot(MDArray* x, MDArray* y) {
    if(x->ndim != 2 || y->ndim != 2) {
        printf("x and/or y ndim is different than 2\n");
        // TODO: If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. https://numpy.org/doc/2.1/reference/generated/numpy.matmul.html#numpy.matmul
        return NULL;
    }

    if(x->shape[1] != y->shape[0]) {
        printf("x.shape[1](%zu) different than y.shape[0](%zu)\n", x->shape[1], y->shape[0]);
        return NULL;
    }

    size_t shape[] = {x->shape[0], y->shape[1]};
    MDArray* out = mdarray_create(2, shape, sizeof(double));
    for(size_t i = 0; i < x->shape[0]; i++) {
        for(size_t k = 0; k < y->shape[1]; k++) {
            double outval = 0;  // Reset for each element of output matrix
            for(size_t j = 0; j < y->shape[0]; j++) {
                size_t ix[] = {i, j};
                size_t iy[] = {j, k};
                double xval = *(double*) mdarray_get_element(x, ix);
                double yval = *(double*) mdarray_get_element(y, iy);
                outval += xval*yval;
            }
            size_t idx[] = {i, k};
            mdarray_set_element(out, idx, &outval);  // Write after full sum is computed
        }
    }

    return out;
}


MDArray* mdarray_copy(MDArray* arr, size_t ndim, size_t* start) {
    MDArray* new_arr = (MDArray*)malloc(sizeof(MDArray));
    if (!new_arr || !arr) return NULL;

    new_arr->ndim = arr->ndim - ndim;
    new_arr->itemsize = arr->itemsize;

    // Allocate and copy shape array
    new_arr->shape = (size_t*)malloc(new_arr->ndim * sizeof(size_t));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }

    memcpy(new_arr->shape, &arr->shape[ndim], new_arr->ndim * sizeof(size_t));

    // Calculate strides
    new_arr->strides = (size_t*)malloc(new_arr->ndim * sizeof(size_t));
    if (!new_arr->strides) {
        free(new_arr->shape);
        free(new_arr);
        return NULL;
    }

    new_arr->total_size = 1;
    for (size_t i = 0; i < new_arr->ndim; i++) {
        new_arr->total_size *= new_arr->shape[i];
    }

    size_t stride = 1;
    for (size_t i = new_arr->ndim - 1; i < new_arr->ndim; i--) {
        new_arr->strides[i] = stride;
        stride *= new_arr->shape[i];
    }

    size_t flat_index = 0;
    for (size_t i = 0; i < ndim; i++) {
        flat_index += start[i] * arr->strides[i]; // Find position in old array
    }

    // Start pointer at given index
    new_arr->data = &arr->data[flat_index];
    if (!new_arr->data) {
        free(new_arr->strides);
        free(new_arr->shape);
        free(new_arr);
        return NULL;
    }

    return new_arr;
}

MDArray* mdarray_resize(MDArray* arr, size_t ndim, size_t* shape) {
    MDArray* new_arr = (MDArray*)malloc(sizeof(MDArray));
    if(!new_arr) return NULL;

    new_arr->ndim = ndim;
    new_arr->itemsize = arr->itemsize;
    new_arr->total_size = arr->total_size;

    // Allocate and copy shape array
    new_arr->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(new_arr->shape, shape, ndim * sizeof(size_t));

    // Calculate strides
    new_arr->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!new_arr->strides) {
        free(new_arr->shape);
        free(new_arr);
        return NULL;
    }

    size_t stride = 1;
    for (size_t i = ndim - 1; i < ndim; i--) {
        new_arr->strides[i] = stride;
        stride *= shape[i];
    }

    new_arr->data = arr->data;

    return new_arr;
}

MDArray* mdarray_sum(MDArray* a, MDArray* b) {
    // TODO: Figure out how to do sum with different shapes check numpy doc.
    MDArray* out = mdarray_create(2, a->shape, sizeof(double));
    for(size_t i = 0; i < a->total_size; i++) {
        double x = *(double*)&a->data[i];
        double y = x + *(double*)&b->data[i];
        memcpy((char*)out->data + (i * a->itemsize), &y, a->itemsize);
    }

    return out;
}



void mdarray_ones(MDArray* arr) {
    for(size_t i = 0; i < arr->total_size; i++) {
        double x = 1;
        memcpy((char*)arr->data + (i * arr->itemsize), &x, arr->itemsize);
    }
}

void mdarray_zeros(MDArray* arr) {
    for(size_t i = 0; i < arr->total_size; i++) {
        double x = 0;
        memcpy((char*)arr->data + (i * arr->itemsize), &x, arr->itemsize);
    }
}