set(LLVM_LINK_COMPONENTS support)
add_clang_executable(transformation3
	T3.cpp
	)
target_link_libraries(transformation3
	clangTooling
	clangBasic
	clangASTMatchers
	)
