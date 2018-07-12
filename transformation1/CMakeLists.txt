set(LLVM_LINK_COMPONENTS support)
add_clang_executable(transformation1
	T1.cpp
	)
target_link_libraries(transformation1
	clangTooling
	clangBasic
	clangASTMatchers
	)
