// mdarray.h
#ifndef MDARRAY_H
#define MDARRAY_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// Structure to hold array metadata
typedef struct {
    void* data;           // Pointer to contiguous data
    size_t* shape;        // Array dimensions
    size_t* strides;      // Number of elements to skip in each dimension
    size_t ndim;          // Number of dimensions
    size_t itemsize;      // Size of each element in bytes
    size_t total_size;    // Total number of elements
} MDArray;

// Function declarations
MDArray* mdarray_create(size_t ndim, size_t* shape, size_t itemsize);
void mdarray_free(MDArray* arr);
void* mdarray_get_element(MDArray* arr, size_t* indices);
void mdarray_set_element(MDArray* arr, size_t* indices, void* value);
size_t mdarray_calculate_index(MDArray* arr, size_t* indices);

// Implementation
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
            printf("Index out of bounds: indices[%zu]=%zu >= shape[%zu]=%zu\n",
                   i, indices[i], i, arr->shape[i]);
            return (size_t)-1;  // Return max value to indicate error
        }
        flat_index += indices[i] * arr->strides[i];
    }
    return flat_index;
}

void* mdarray_get_element(MDArray* arr, size_t* indices) {
    size_t index = mdarray_calculate_index(arr, indices);
    return (char*)arr->data + (index * arr->itemsize);
}

void mdarray_set_element(MDArray* arr, size_t* indices, void* value) {
    size_t index = mdarray_calculate_index(arr, indices);
    memcpy((char*)arr->data + (index * arr->itemsize), value, arr->itemsize);
}

#endif // MDARRAY_H
