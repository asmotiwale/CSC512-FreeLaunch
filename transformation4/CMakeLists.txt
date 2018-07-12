set(LLVM_LINK_COMPONENTS support)
add_clang_executable(transformation4
	T4.cpp
	)
target_link_libraries(transformation4
	clangTooling
	clangBasic
	clangASTMatchers
	)
