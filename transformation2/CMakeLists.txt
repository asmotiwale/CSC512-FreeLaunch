set(LLVM_LINK_COMPONENTS support)
add_clang_executable(transformation2
	T2.cpp
	)
target_link_libraries(transformation2
	clangTooling
	clangBasic
	clangASTMatchers
	)
