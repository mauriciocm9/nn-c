#include "unity.h"
#include "mdarray.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

// Helper function for floating point comparison
#define FLOAT_EPSILON 0.0001f
static int float_eq(double a, double b) {
    return fabs(a - b) < FLOAT_EPSILON;
}

void test_mdarray_creation_and_access(void) {
    // Test creating a 2D array
    size_t shape[] = {2, 3};
    MDArray* arr = mdarray_create(2, shape, sizeof(double));

    // Verify array properties
    TEST_ASSERT_NOT_NULL(arr);
    TEST_ASSERT_EQUAL(2, arr->ndim);
    TEST_ASSERT_EQUAL(6, arr->total_size);
    TEST_ASSERT_EQUAL(sizeof(double), arr->itemsize);

    // Test setting and getting elements
    double value = 42.0;
    size_t indices[] = {1, 2};
    mdarray_set_element(arr, indices, &value);

    double* retrieved = (double*)mdarray_get_element(arr, indices);
    TEST_ASSERT_TRUE(float_eq(value, *retrieved));

    // Test bounds checking
    size_t invalid_indices[] = {2, 3}; // Out of bounds
    void* result = mdarray_get_element(arr, invalid_indices);
    TEST_ASSERT_NULL(result);

    mdarray_free(arr);
}

void test_mdarray_dot_product(void) {
    // Create two matrices for multiplication
    // Matrix A: 2x3
    size_t shape_a[] = {2, 3};
    MDArray* a = mdarray_create(2, shape_a, sizeof(double));

    // Matrix B: 3x2
    size_t shape_b[] = {3, 2};
    MDArray* b = mdarray_create(2, shape_b, sizeof(double));

    // Fill matrix A with values
    double val_a;
    for(size_t i = 0; i < 2; i++) {
        for(size_t j = 0; j < 3; j++) {
            val_a = i * 3 + j + 1; // Values: 1,2,3,4,5,6
            size_t indices[] = {i, j};
            mdarray_set_element(a, indices, &val_a);
        }
    }

    // Fill matrix B with values
    double val_b;
    for(size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < 2; j++) {
            val_b = i * 2 + j + 1; // Values: 1,2,3,4,5,6
            size_t indices[] = {i, j};
            mdarray_set_element(b, indices, &val_b);
        }
    }

    // Perform matrix multiplication
    MDArray* result = mdarray_dot(a, b);

    // Verify result dimensions
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);

    // Expected results:
    // [1 2 3]   [1 2]   [22 28]
    // [4 5 6] * [3 4] = [49 64]
    //           [5 6]

    size_t idx00[] = {0, 0};
    size_t idx01[] = {0, 1};
    size_t idx10[] = {1, 0};
    size_t idx11[] = {1, 1};

    double* val00 = (double*)mdarray_get_element(result, idx00);
    double* val01 = (double*)mdarray_get_element(result, idx01);
    double* val10 = (double*)mdarray_get_element(result, idx10);
    double* val11 = (double*)mdarray_get_element(result, idx11);

    TEST_ASSERT_TRUE(float_eq(22.0, *val00));
    TEST_ASSERT_TRUE(float_eq(28.0, *val01));
    TEST_ASSERT_TRUE(float_eq(49.0, *val10));
    TEST_ASSERT_TRUE(float_eq(64.0, *val11));

    mdarray_free(a);
    mdarray_free(b);
    mdarray_free(result);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mdarray_creation_and_access);
    RUN_TEST(test_mdarray_dot_product);
    return UNITY_END();
}