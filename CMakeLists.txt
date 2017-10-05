cmake_minimum_required(VERSION 2.8)

PROJECT (SfmopenMVG)

find_package(OpenCV  REQUIRED)
find_package(OpenMVG  REQUIRED)
find_package(Ceres  REQUIRED)
find_package(VTK  OPTIONAL_COMPONENTS)

file(GLOB SfmopenMVG_SRCS
    "*.h"
    "*.cpp")
if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

get_target_property(OpenMVG OpenMVG INTERFACE_LINK_LIBRARIES)

ADD_EXECUTABLE (SfmopenMVG ${SfmopenMVG_SRCS})



if (OpenCV_FOUND AND OPENMVG_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} ${OPENMVG_INCLUDE_DIRS} )
      if (MSVC)
	message ("MSVC configuration")
      add_definitions( -D_USE_MATH_DEFINES  -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS  )
	target_link_libraries( SfmopenMVG debug  ${OpenCV_LIBS} ${Ceres_LIBS} ${OPENMVG_LIBRARIES} )
	target_link_libraries( SfmopenMVG optimized ${OpenCV_LIBS} ${Ceres_LIBS} ${OPENMVG_LIBRARIES} )
	else()
	target_link_libraries( SfmopenMVG ${OpenCV_LIBS} ${Ceres_LIBS} ${OPENMVG_LIBRARIES} )
	endif()
	if (VTK_FOUND)
		add_definitions(-DUSE_VTK)
		include_directories( ${VTK_INCLUDE_DIRS} )
	endif (Vtk_FOUND)
else ()
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif ()

