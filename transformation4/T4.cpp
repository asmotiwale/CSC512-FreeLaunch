/*
Author: 1. Anuraag Motiwale   asmotiwa@ncsu.edu
        2. Abhishek Singh     aksingh5@ncsu.edu
*/
#include <string>
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;


static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class cudaSubkernelCallHandler : public MatchFinder::MatchCallback {
    public:
          cudaSubkernelCallHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

            virtual void run(const MatchFinder::MatchResult &Result) {
                    const CUDAKernelCallExpr *subKernelCall = Result.Nodes.getNodeAs<CUDAKernelCallExpr>("subKernelCall");
                const FunctionDecl *kernelFunctionDecl = Result.Nodes.getNodeAs<FunctionDecl>("cudaKernelCallDecl");
                const CUDAKernelCallExpr *cudaKernelCall = Result.Nodes.getNodeAs<CUDAKernelCallExpr>("cudaKernelCall");
                const Stmt *kernelCallBody = kernelFunctionDecl->getBody();
                const FunctionDecl *childfunction = subKernelCall->getDirectCallee();
                const Stmt *childCallBody = subKernelCall->getCalleeDecl()->getBody();
                const CallExpr *subkernelCallPreArg= subKernelCall->getConfig();
                const CallExpr *kernelCallPreArg= cudaKernelCall->getConfig();
                const NamedDecl *childTargetDecl = Result.Nodes.getNodeAs<NamedDecl>("targetexpression");
                //const Expr *temp=childTargetDecl->getLHS();

                clang::SourceManager &SM=Result.Context->getSourceManager();
                clang::LangOptions LangOpts;
                clang::PrintingPolicy Policy(LangOpts);
                std::map<std::string,std::string> subkernelCallParams;
                std::string forIdentifier = childTargetDecl->getName();



                //Extract the numberofblocks and numberofblockthreads enclosed within <<<>>> for both kernel and subkernel calls.
                std::string subKernelCallpreArgs[2];
                                for(int i=0;i<2;i++){
                    std::string argument;
                    llvm::raw_string_ostream arg(argument);
                    subkernelCallPreArg->getArg(i)->printPretty(arg, 0, Policy);
                    subKernelCallpreArgs[i]=arg.str();
                }

                std::string kernelCallpreArgs[2];
                for(int i=0;i<2;i++){
                        std::string argument;
                        llvm::raw_string_ostream arg(argument);
                        kernelCallPreArg->getArg(i)->printPretty(arg, 0, Policy);
                        kernelCallpreArgs[i]=arg.str();
                }

                std::string childKernelParam = "\nint FL_childGridThreads = int("+subKernelCallpreArgs[0]+")*"+subKernelCallpreArgs[1]+";\n";
                //Rewrite.InsertText(subKernelCall->getLocStart(), childKernelParam, true, true);


                //String for for loop for child kernel call
                std::string forLoopStmt = "for(unsigned "+forIdentifier+"0; "+forIdentifier+" < FL_childGridThreads; ++"+forIdentifier+"){";

                //Logic to copy child kernel function body into kernel function body
                llvm::StringRef childbodytext = Lexer::getSourceText(CharSourceRange::getTokenRange(SourceRange(childCallBody->getLocStart(),childCallBody->getLocEnd())),SM, LangOpts);
                //get rid of enclosing {}
                std::string strTrimmed= childbodytext.str().substr(2,childbodytext.str().length()-2);

                //Replace the child thread id calculation expression.
                replaceAll(strTrimmed,"threadIdx.x + blockIdx.x*blockDim.x;"," 0; "+forIdentifier+" < FL_childGridThreads; ++"+forIdentifier+"){");
                replaceAll(strTrimmed,"threadIdx.x+ blockIdx.x*blockDim.x;"," 0; "+forIdentifier+" < FL_childGridThreads; ++"+forIdentifier+"){");
                replaceAll(strTrimmed,"threadIdx.x +blockIdx.x*blockDim.x;"," 0; "+forIdentifier+" < FL_childGridThreads; ++"+forIdentifier+"){");
                replaceAll(strTrimmed,"threadIdx.x+blockIdx.x*blockDim.x;"," 0; "+forIdentifier+" < FL_childGridThreads; ++"+forIdentifier+"){");


                //Replace the subkernel call expression with above assembled expression.
                Rewrite.ReplaceText(SourceRange(subKernelCall->getLocStart(),subKernelCall->getLocEnd()),childKernelParam+"\n"+""+"\n"+"for("+strTrimmed);

                //Remove child function
                SourceLocation startLoc = SM.getFileLoc(childfunction->getLocStart());
                SourceLocation endLoc = SM.getFileLoc(childfunction->getLocEnd());
                Rewrite.ReplaceText(SourceRange(startLoc,endLoc),"");
            }



            void replaceAll(std::string& str, const std::string& from, const std::string& to) {
                if(from.empty())
                    return;
                size_t start_pos = 0;
                while((start_pos = str.find(from, start_pos)) != std::string::npos) {
                    str.replace(start_pos, from.length(), to);
                    start_pos += to.length();
                }
            }

            static std::string getText(const SourceManager &SourceManager, const Expr &Node) {
                         SourceLocation StartSpellingLocation = SourceManager.getSpellingLoc(Node.getLocStart());
                         SourceLocation EndSpellingLocation = SourceManager.getSpellingLoc(Node.getLocEnd());
                         if (!StartSpellingLocation.isValid() || !EndSpellingLocation.isValid()) {
                                return std::string();
                         }
                         bool Invalid = true;
                         const char *Text = SourceManager.getCharacterData(StartSpellingLocation, &Invalid);
                         if (Invalid) {
                                return std::string();
                         }
                         std::pair<FileID, unsigned> Start = SourceManager.getDecomposedLoc(StartSpellingLocation);
                         std::pair<FileID, unsigned> End = SourceManager.getDecomposedLoc(Lexer::getLocForEndOfToken(EndSpellingLocation, 0, SourceManager, LangOptions()));
                         if (Start.first != End.first) {
                                return std::string();
                         }
                         if (End.second < Start.second) {
                                return std::string();
                         }
                         return std::string(Text, End.second - Start.second);
                 }

    private:
              Rewriter &Rewrite;
};



class cudaSubKernelReturnCallHandler : public MatchFinder::MatchCallback {
        public:
                     cudaSubKernelReturnCallHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

                      virtual void run(const MatchFinder::MatchResult &Result) {
                     const ReturnStmt *returnstmt= Result.Nodes.getNodeAs<ReturnStmt>("returnstmt");
                 //replace all return statements in kernel function body with goto P.
                 Rewrite.ReplaceText(SourceRange(returnstmt->getLocStart(),returnstmt->getLocEnd()),"return");
              }
        private:
              Rewriter &Rewrite;

};


class cudaSubKernelBlockIdExprHandler : public MatchFinder::MatchCallback {
        public:
                     cudaSubKernelBlockIdExprHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

                     virtual void run(const MatchFinder::MatchResult &Result) {
                 //logic to alter the thread id calculation in kernel module.
                         const BinaryOperator *blockstmt= Result.Nodes.getNodeAs<BinaryOperator>("targetexpr");
                         const Expr *blockstmtlhs=blockstmt->getLHS();
                 clang::SourceManager &SM=Result.Context->getSourceManager();
                 std::string blockoperand=getText(SM,*blockstmtlhs);

                     }

             //function to extract expr as string.
                 static std::string getText(const SourceManager &SourceManager, const Expr &Node) {
                         SourceLocation StartSpellingLocation = SourceManager.getSpellingLoc(Node.getLocStart());
                         SourceLocation EndSpellingLocation = SourceManager.getSpellingLoc(Node.getLocEnd());
                         if (!StartSpellingLocation.isValid() || !EndSpellingLocation.isValid()) {
                                return std::string();
                         }
                         bool Invalid = true;
                         const char *Text = SourceManager.getCharacterData(StartSpellingLocation, &Invalid);
                         if (Invalid) {
                                return std::string();
                         }
                         std::pair<FileID, unsigned> Start = SourceManager.getDecomposedLoc(StartSpellingLocation);
                         std::pair<FileID, unsigned> End = SourceManager.getDecomposedLoc(Lexer::getLocForEndOfToken(EndSpellingLocation, 0, SourceManager, LangOptions()));
                         if (Start.first != End.first) {
                                return std::string();
                         }
                         if (End.second < Start.second) {
                                return std::string();
                         }
                         return std::string(Text, End.second - Start.second);
                 }

    private:
                     Rewriter &Rewrite;

};


class MyASTConsumer : public ASTConsumer {
    public:
        MyASTConsumer(Rewriter &R) : HandlerForCudaSubKernCall(R), HandlerForCudaSubKernReturnCall(R), HandlerForcudaSubKernelBlockIdExpr(R) {
                Matcher.addMatcher(cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr()),forEachDescendant(returnStmt().bind("returnstmt"))))), &HandlerForCudaSubKernReturnCall);

                Matcher.addMatcher(cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr(callee(functionDecl(hasDescendant(varDecl(hasType(isUnsignedInteger()),hasInitializer(expr(binaryOperator(hasOperatorName("+"),hasRHS(expr(binaryOperator(hasOperatorName("*")))))))).bind("targetexpression"))))).bind("subKernelCall"))).bind("cudaKernelCallDecl"))).bind("cudaKernelCall"), &HandlerForCudaSubKernCall);


                Matcher.addMatcher(cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr()),hasDescendant(varDecl(hasType(isUnsignedInteger()),hasInitializer(expr(binaryOperator(hasOperatorName("+"),hasLHS(expr(binaryOperator(hasOperatorName("*"))).bind("targetexpr")))))))))), &HandlerForcudaSubKernelBlockIdExpr);

        }


        void HandleTranslationUnit(ASTContext &Context) override {
                Matcher.matchAST(Context);
        }

private:
        cudaSubkernelCallHandler HandlerForCudaSubKernCall;
        cudaSubKernelReturnCallHandler HandlerForCudaSubKernReturnCall;
        cudaSubKernelBlockIdExprHandler HandlerForcudaSubKernelBlockIdExpr;
        MatchFinder Matcher;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
      .write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                         StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
