#add_library(mdarray
    #mdarray.c
    #)

#target_include_directories(mdarray
#    PUBLIC
#        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
#        $<INSTALL_INTERFACE:include>
#    PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}
#)

#set_target_properties(mdarray PROPERTIES
#VERSION ${PROJECT_VERSION}
#SOVERSION ${PROJECT_VERSION_MAJOR}
#)

#install(TARGETS mdarray
#    EXPORT mdarrayTargets
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#)

#add_executable(main main.c)
#target_link_libraries(main PRIVATE mdarray)

# List of source files
cmake_minimum_required(VERSION 3.10)

set(SOURCE_FILES
    main.c
    mdarray.c
)

add_executable(main ${SOURCE_FILES})
