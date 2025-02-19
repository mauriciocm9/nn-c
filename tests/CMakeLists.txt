add_executable(_tests
        unity/src/unity.c   # Unity framework
        test_mdarray.c
)

# Include Unity headers
target_include_directories(_tests PRIVATE unity/src)
target_include_directories(_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unity/src)

# Enable testing
enable_testing()
add_test(NAME RunUnitTests COMMAND _tests)