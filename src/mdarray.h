// mdarray.h
#ifndef MDARRAY_H
#define MDARRAY_H

#include <stdio.h>

// Structure to hold array metadata
typedef struct {
    void* data;           // Pointer to contiguous data
    size_t* shape;        // Array dimensions
    size_t* strides;      // Number of elements to skip in each dimension
    size_t ndim;          // Number of dimensions
    size_t itemsize;      // Size of each element in bytes
    size_t total_size;    // Total number of elements
} MDArray;

MDArray* mdarray_create(size_t ndim, size_t* shape, size_t itemsize);
void mdarray_free(MDArray* arr);
void* mdarray_get_element(MDArray* arr, size_t* indices);
void mdarray_set_element(MDArray* arr, size_t* indices, void* value);
size_t mdarray_calculate_index(MDArray* arr, size_t* indices);
MDArray* mdarray_dot(MDArray* x, MDArray* y);
void mdarray_ones(MDArray* arr);
void mdarray_zeros(MDArray* arr);
MDArray* mdarray_resize(MDArray* arr, size_t ndim, size_t* shape);
MDArray* mdarray_copy(MDArray* arr, size_t ndim, size_t* start);
MDArray* mdarray_sum(MDArray* a, MDArray* b);

#endif // MDARRAY_H
