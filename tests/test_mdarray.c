#include "unity.h"

void setUp (void) {} /* Is run before every test, put unit init calls here. */
void tearDown (void) {} /* Is run after every test, put unit clean-up calls here. */

int8_t AverageThreeBytes(int8_t a, int8_t b, int8_t c)
{
    return (int8_t)(((int16_t)a + (int16_t)b + (int16_t)c) / 3);
}

void test_AverageThreeBytes_should_AverageMidRangeValues(void)
{
    TEST_ASSERT_EQUAL_HEX8(40, AverageThreeBytes(30, 40, 50));
    TEST_ASSERT_EQUAL_HEX8(40, AverageThreeBytes(10, 70, 40));
    TEST_ASSERT_EQUAL_HEX8(33, AverageThreeBytes(33, 33, 33));
}

void test_AverageThreeBytes_should_AverageHighValues(void)
{
    TEST_ASSERT_EQUAL_HEX8(80, AverageThreeBytes(70, 80, 90));
    TEST_ASSERT_EQUAL_HEX8(127, AverageThreeBytes(127, 127, 127));
    TEST_ASSERT_EQUAL_HEX8(84, AverageThreeBytes(0, 126, 126));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_AverageThreeBytes_should_AverageMidRangeValues);
    RUN_TEST(test_AverageThreeBytes_should_AverageHighValues);
    return UNITY_END();
}