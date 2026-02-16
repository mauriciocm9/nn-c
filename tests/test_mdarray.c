#include "unity.h"
#include "mdarray.h"
#include "linear.h"
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

void test_mdarray_get_2d_returns_1d_view(void) {
    // Create a 2x3 array: [[1,2,3],[4,5,6]]
    size_t shape[] = {2, 3};
    MDArray* arr = mdarray_create(2, shape, sizeof(double));
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            double val = i * 3 + j + 1;
            size_t idx[] = {i, j};
            mdarray_set_element(arr, idx, &val);
        }
    }

    MDArray* row0 = mdarray_get(arr, 0);
    TEST_ASSERT_NOT_NULL(row0);
    TEST_ASSERT_EQUAL(1, row0->ndim);
    TEST_ASSERT_EQUAL(3, row0->shape[0]);
    TEST_ASSERT_EQUAL(3, row0->total_size);
    TEST_ASSERT_FALSE(row0->owns_data);

    size_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    TEST_ASSERT_TRUE(float_eq(1.0, *(double*)mdarray_get_element(row0, i0)));
    TEST_ASSERT_TRUE(float_eq(2.0, *(double*)mdarray_get_element(row0, i1)));
    TEST_ASSERT_TRUE(float_eq(3.0, *(double*)mdarray_get_element(row0, i2)));

    MDArray* row1 = mdarray_get(arr, 1);
    TEST_ASSERT_NOT_NULL(row1);
    TEST_ASSERT_TRUE(float_eq(4.0, *(double*)mdarray_get_element(row1, i0)));
    TEST_ASSERT_TRUE(float_eq(5.0, *(double*)mdarray_get_element(row1, i1)));
    TEST_ASSERT_TRUE(float_eq(6.0, *(double*)mdarray_get_element(row1, i2)));

    mdarray_free(row0);
    mdarray_free(row1);
    mdarray_free(arr);
}

void test_mdarray_get_3d_returns_2d_view(void) {
    // Create a 2x3x4 array
    size_t shape[] = {2, 3, 4};
    MDArray* arr = mdarray_create(3, shape, sizeof(double));
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 4; k++) {
                double val = i * 12 + j * 4 + k;
                size_t idx[] = {i, j, k};
                mdarray_set_element(arr, idx, &val);
            }
        }
    }

    MDArray* slice = mdarray_get(arr, 1);
    TEST_ASSERT_NOT_NULL(slice);
    TEST_ASSERT_EQUAL(2, slice->ndim);
    TEST_ASSERT_EQUAL(3, slice->shape[0]);
    TEST_ASSERT_EQUAL(4, slice->shape[1]);
    TEST_ASSERT_EQUAL(12, slice->total_size);

    // slice[0][0] should be arr[1][0][0] = 12
    size_t idx[] = {0, 0};
    TEST_ASSERT_TRUE(float_eq(12.0, *(double*)mdarray_get_element(slice, idx)));

    // slice[2][3] should be arr[1][2][3] = 23
    size_t idx2[] = {2, 3};
    TEST_ASSERT_TRUE(float_eq(23.0, *(double*)mdarray_get_element(slice, idx2)));

    mdarray_free(slice);
    mdarray_free(arr);
}

void test_mdarray_get_chained(void) {
    // Create a 2x3x4 array and do get(get(arr, 1), 2)
    size_t shape[] = {2, 3, 4};
    MDArray* arr = mdarray_create(3, shape, sizeof(double));
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 4; k++) {
                double val = i * 12 + j * 4 + k;
                size_t idx[] = {i, j, k};
                mdarray_set_element(arr, idx, &val);
            }
        }
    }

    MDArray* slice = mdarray_get(arr, 1);
    MDArray* row = mdarray_get(slice, 2);
    TEST_ASSERT_NOT_NULL(row);
    TEST_ASSERT_EQUAL(1, row->ndim);
    TEST_ASSERT_EQUAL(4, row->shape[0]);

    // row[0] = arr[1][2][0] = 20
    size_t i0[] = {0};
    TEST_ASSERT_TRUE(float_eq(20.0, *(double*)mdarray_get_element(row, i0)));
    // row[3] = arr[1][2][3] = 23
    size_t i3[] = {3};
    TEST_ASSERT_TRUE(float_eq(23.0, *(double*)mdarray_get_element(row, i3)));

    mdarray_free(row);
    mdarray_free(slice);
    mdarray_free(arr);
}

void test_mdarray_get_1d_returns_null(void) {
    size_t shape[] = {5};
    MDArray* arr = mdarray_create(1, shape, sizeof(double));
    TEST_ASSERT_NULL(mdarray_get(arr, 0));
    mdarray_free(arr);
}

void test_mdarray_get_out_of_bounds_returns_null(void) {
    size_t shape[] = {2, 3};
    MDArray* arr = mdarray_create(2, shape, sizeof(double));
    TEST_ASSERT_NULL(mdarray_get(arr, 2));
    TEST_ASSERT_NULL(mdarray_get(arr, 100));
    mdarray_free(arr);
}

void test_svm_loss_some_margins_violated(void) {
    // scores: [2.0, 5.0, 3.0] with correct_class=1 (score=5.0)
    // j=0: max(0, 2.0 - 5.0 + 1.0) = 0.0
    // j=2: max(0, 3.0 - 5.0 + 1.0) = 0.0
    // loss = 0.0
    size_t shape[] = {3, 1};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));
    double v0 = 2.0, v1 = 5.0, v2 = 3.0;
    size_t i0[] = {0, 0}; mdarray_set_element(scores, i0, &v0);
    size_t i1[] = {1, 0}; mdarray_set_element(scores, i1, &v1);
    size_t i2[] = {2, 0}; mdarray_set_element(scores, i2, &v2);

    size_t label = 1;
    double loss = svm_loss(scores, &label, 1);
    TEST_ASSERT_TRUE(float_eq(0.0, loss));

    // Now correct_class=0 (score=2.0)
    // j=1: max(0, 5.0 - 2.0 + 1.0) = 4.0
    // j=2: max(0, 3.0 - 2.0 + 1.0) = 2.0
    // loss = 6.0
    label = 0;
    loss = svm_loss(scores, &label, 1);
    TEST_ASSERT_TRUE(float_eq(6.0, loss));

    mdarray_free(scores);
}

void test_svm_loss_correct_class_highest(void) {
    // scores: [1.0, 10.0] with correct_class=1
    // j=0: max(0, 1.0 - 10.0 + 1.0) = 0.0
    size_t shape[] = {2, 1};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));
    double v0 = 1.0, v1 = 10.0;
    size_t i0[] = {0, 0}; mdarray_set_element(scores, i0, &v0);
    size_t i1[] = {1, 0}; mdarray_set_element(scores, i1, &v1);

    size_t label = 1;
    double loss = svm_loss(scores, &label, 1);
    TEST_ASSERT_TRUE(float_eq(0.0, loss));

    mdarray_free(scores);
}

void test_svm_loss_all_equal_scores(void) {
    // All scores = 3.0, correct_class=0, 4 classes
    // Each j!=0: max(0, 3.0 - 3.0 + 1.0) = 1.0
    // loss = 3 * 1.0 = 3.0 = (num_classes - 1) * delta
    size_t shape[] = {4, 1};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));
    double v = 3.0;
    for (size_t i = 0; i < 4; i++) {
        size_t idx[] = {i, 0};
        mdarray_set_element(scores, idx, &v);
    }

    size_t label = 0;
    double loss = svm_loss(scores, &label, 1);
    TEST_ASSERT_TRUE(float_eq(3.0, loss));

    mdarray_free(scores);
}

void test_svm_loss_batch(void) {
    // 3 classes, batch of 2 → scores shape (3, 2)
    // Sample 0: scores [2.0, 5.0, 3.0], label=1 → loss=0.0
    // Sample 1: scores [4.0, 1.0, 2.0], label=0
    //   j=1: max(0, 1.0 - 4.0 + 1.0) = 0.0
    //   j=2: max(0, 2.0 - 4.0 + 1.0) = 0.0
    //   → loss=0.0
    // mean = (0.0 + 0.0) / 2 = 0.0
    size_t shape[] = {3, 2};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));

    // Column 0: [2.0, 5.0, 3.0]
    double vals[6] = {2.0, 4.0, 5.0, 1.0, 3.0, 2.0};
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 2; c++) {
            size_t idx[] = {r, c};
            mdarray_set_element(scores, idx, &vals[r * 2 + c]);
        }
    }

    size_t labels[] = {1, 0};
    double loss = svm_loss(scores, labels, 2);
    TEST_ASSERT_TRUE(float_eq(0.0, loss));

    // Now change labels so margins are violated
    // Sample 0: label=0 (score=2.0)
    //   j=1: max(0, 5.0-2.0+1.0) = 4.0
    //   j=2: max(0, 3.0-2.0+1.0) = 2.0  → 6.0
    // Sample 1: label=2 (score=2.0)
    //   j=0: max(0, 4.0-2.0+1.0) = 3.0
    //   j=1: max(0, 1.0-2.0+1.0) = 0.0  → 3.0
    // mean = (6.0 + 3.0) / 2 = 4.5
    size_t labels2[] = {0, 2};
    loss = svm_loss(scores, labels2, 2);
    TEST_ASSERT_TRUE(float_eq(4.5, loss));

    mdarray_free(scores);
}

void test_svm_loss_backward_shape_and_values(void) {
    // 3 classes, batch=1, scores: [2.0, 5.0, 3.0], label=0 (score=2.0)
    // j=1: margin = 5.0 - 2.0 + 1.0 = 4.0 > 0 → dscores[1,0] = 1/1 = 1.0
    // j=2: margin = 3.0 - 2.0 + 1.0 = 2.0 > 0 → dscores[2,0] = 1/1 = 1.0
    // correct class j=0: count=2 → dscores[0,0] = -2/1 = -2.0
    size_t shape[] = {3, 1};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));
    double v0 = 2.0, v1 = 5.0, v2 = 3.0;
    size_t i0[] = {0, 0}; mdarray_set_element(scores, i0, &v0);
    size_t i1[] = {1, 0}; mdarray_set_element(scores, i1, &v1);
    size_t i2[] = {2, 0}; mdarray_set_element(scores, i2, &v2);

    size_t label = 0;
    MDArray* dscores = svm_loss_backward(scores, &label, 1);

    TEST_ASSERT_NOT_NULL(dscores);
    TEST_ASSERT_EQUAL(2, dscores->ndim);
    TEST_ASSERT_EQUAL(3, dscores->shape[0]);
    TEST_ASSERT_EQUAL(1, dscores->shape[1]);

    size_t d0[] = {0, 0}, d1[] = {1, 0}, d2[] = {2, 0};
    TEST_ASSERT_TRUE(float_eq(-2.0, *(double*)mdarray_get_element(dscores, d0)));
    TEST_ASSERT_TRUE(float_eq(1.0, *(double*)mdarray_get_element(dscores, d1)));
    TEST_ASSERT_TRUE(float_eq(1.0, *(double*)mdarray_get_element(dscores, d2)));

    mdarray_free(scores);
    mdarray_free(dscores);
}

void test_svm_loss_backward_no_violation(void) {
    // 3 classes, batch=1, scores: [2.0, 5.0, 3.0], label=1 (score=5.0)
    // j=0: margin = 2.0 - 5.0 + 1.0 = -2.0 ≤ 0 → 0
    // j=2: margin = 3.0 - 5.0 + 1.0 = -1.0 ≤ 0 → 0
    // correct class j=1: count=0 → dscores[1,0] = 0.0
    size_t shape[] = {3, 1};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));
    double v0 = 2.0, v1 = 5.0, v2 = 3.0;
    size_t i0[] = {0, 0}; mdarray_set_element(scores, i0, &v0);
    size_t i1[] = {1, 0}; mdarray_set_element(scores, i1, &v1);
    size_t i2[] = {2, 0}; mdarray_set_element(scores, i2, &v2);

    size_t label = 1;
    MDArray* dscores = svm_loss_backward(scores, &label, 1);

    size_t d0[] = {0, 0}, d1[] = {1, 0}, d2[] = {2, 0};
    TEST_ASSERT_TRUE(float_eq(0.0, *(double*)mdarray_get_element(dscores, d0)));
    TEST_ASSERT_TRUE(float_eq(0.0, *(double*)mdarray_get_element(dscores, d1)));
    TEST_ASSERT_TRUE(float_eq(0.0, *(double*)mdarray_get_element(dscores, d2)));

    mdarray_free(scores);
    mdarray_free(dscores);
}

void test_svm_loss_backward_batch(void) {
    // 3 classes, batch=2
    // Sample 0: scores [2.0, 5.0, 3.0], label=0
    //   j=1: 5-2+1=4>0 → 1/2=0.5; j=2: 3-2+1=2>0 → 0.5; correct: -2/2=-1.0
    // Sample 1: scores [4.0, 1.0, 2.0], label=2
    //   j=0: 4-2+1=3>0 → 0.5; j=1: 1-2+1=0≤0 → 0; correct: -1/2=-0.5
    size_t shape[] = {3, 2};
    MDArray* scores = mdarray_create(2, shape, sizeof(double));
    double vals[6] = {2.0, 4.0, 5.0, 1.0, 3.0, 2.0};
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 2; c++) {
            size_t idx[] = {r, c};
            mdarray_set_element(scores, idx, &vals[r * 2 + c]);
        }
    }

    size_t labels[] = {0, 2};
    MDArray* dscores = svm_loss_backward(scores, labels, 2);

    size_t d00[] = {0, 0}, d10[] = {1, 0}, d20[] = {2, 0};
    size_t d01[] = {0, 1}, d11[] = {1, 1}, d21[] = {2, 1};
    TEST_ASSERT_TRUE(float_eq(-1.0, *(double*)mdarray_get_element(dscores, d00)));
    TEST_ASSERT_TRUE(float_eq(0.5, *(double*)mdarray_get_element(dscores, d10)));
    TEST_ASSERT_TRUE(float_eq(0.5, *(double*)mdarray_get_element(dscores, d20)));
    TEST_ASSERT_TRUE(float_eq(0.5, *(double*)mdarray_get_element(dscores, d01)));
    TEST_ASSERT_TRUE(float_eq(0.0, *(double*)mdarray_get_element(dscores, d11)));
    TEST_ASSERT_TRUE(float_eq(-0.5, *(double*)mdarray_get_element(dscores, d21)));

    mdarray_free(scores);
    mdarray_free(dscores);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mdarray_creation_and_access);
    RUN_TEST(test_mdarray_dot_product);
    RUN_TEST(test_mdarray_get_2d_returns_1d_view);
    RUN_TEST(test_mdarray_get_3d_returns_2d_view);
    RUN_TEST(test_mdarray_get_chained);
    RUN_TEST(test_mdarray_get_1d_returns_null);
    RUN_TEST(test_mdarray_get_out_of_bounds_returns_null);
    RUN_TEST(test_svm_loss_some_margins_violated);
    RUN_TEST(test_svm_loss_correct_class_highest);
    RUN_TEST(test_svm_loss_all_equal_scores);
    RUN_TEST(test_svm_loss_batch);
    RUN_TEST(test_svm_loss_backward_shape_and_values);
    RUN_TEST(test_svm_loss_backward_no_violation);
    RUN_TEST(test_svm_loss_backward_batch);
    return UNITY_END();
}