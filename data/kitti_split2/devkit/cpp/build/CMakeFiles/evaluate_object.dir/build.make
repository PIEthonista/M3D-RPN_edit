# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/u5832291/cmake/bin/cmake

# The command to remove a file.
RM = /home/u5832291/cmake/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/evaluate_object.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/evaluate_object.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/evaluate_object.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/evaluate_object.dir/flags.make

CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o: CMakeFiles/evaluate_object.dir/flags.make
CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o: ../evaluate_object.cpp
CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o: CMakeFiles/evaluate_object.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o -MF CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o.d -o CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o -c /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/evaluate_object.cpp

CMakeFiles/evaluate_object.dir/evaluate_object.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evaluate_object.dir/evaluate_object.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/evaluate_object.cpp > CMakeFiles/evaluate_object.dir/evaluate_object.cpp.i

CMakeFiles/evaluate_object.dir/evaluate_object.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evaluate_object.dir/evaluate_object.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/evaluate_object.cpp -o CMakeFiles/evaluate_object.dir/evaluate_object.cpp.s

# Object files for target evaluate_object
evaluate_object_OBJECTS = \
"CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o"

# External object files for target evaluate_object
evaluate_object_EXTERNAL_OBJECTS =

evaluate_object: CMakeFiles/evaluate_object.dir/evaluate_object.cpp.o
evaluate_object: CMakeFiles/evaluate_object.dir/build.make
evaluate_object: CMakeFiles/evaluate_object.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable evaluate_object"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/evaluate_object.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/evaluate_object.dir/build: evaluate_object
.PHONY : CMakeFiles/evaluate_object.dir/build

CMakeFiles/evaluate_object.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/evaluate_object.dir/cmake_clean.cmake
.PHONY : CMakeFiles/evaluate_object.dir/clean

CMakeFiles/evaluate_object.dir/depend:
	cd /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build /work/u5832291/yixian/M3D_RPN_edit/data/kitti_split2/devkit/cpp/build/CMakeFiles/evaluate_object.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/evaluate_object.dir/depend

