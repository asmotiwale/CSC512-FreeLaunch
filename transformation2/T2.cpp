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

                clang::SourceManager &SM=Result.Context->getSourceManager();
                clang::LangOptions LangOpts;
                clang::PrintingPolicy Policy(LangOpts);
                std::map<std::string,std::string> subkernelCallParams;

                //Insert header.
                Rewrite.InsertText(SM.getLocForStartOfFile(SM.getMainFileID()), "\n#include \"freeLaunch_T2.h\"\n");
                Rewrite.InsertText(SM.getLocForStartOfFile(SM.getMainFileID()), "\n#define _Bool bool\n");


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

                //Add declaration for freelaunch argument array, edit the kernel call to consume additional arguments.
                Rewrite.InsertText(cudaKernelCall->getLocStart(),"char *FL_Arguments; cudaMalloc((void **)&FL_Arguments,MAX_FL_ARGSZ);\ncudaMemset(FL_Arguments,0,MAX_FL_ARGSZ);\n", true, true);
                Rewrite.InsertText(cudaKernelCall->getLocEnd(),","+kernelCallpreArgs[0]+",FL_Arguments", true, true);

                Rewrite.InsertTextAfterToken(cudaKernelCall->getLocEnd(), ";\ncudaFree(FL_Arguments)");

                // Extract [argumentName -> argumentType] pair into a map for subkernel call.
                for(int i=0, j=subKernelCall->getNumArgs();i<j;i++){
                        std::string argument;
                        llvm::raw_string_ostream arg(argument);
                    subKernelCall->getArg(i)->printPretty(arg, 0, Policy);
                    subkernelCallParams[arg.str()]=subKernelCall->getArg(i)->getType().getAsString();
                }

                // Remove [argumentName -> argumentType] pairs from map which we are already getting for kernel call as parameters.
                std::map<std::string,std::string>::iterator it;
                for(int i=0, j=kernelFunctionDecl->getNumParams();i<j;i++){
                    it=subkernelCallParams.find(kernelFunctionDecl->parameters()[i]->getQualifiedNameAsString());
                    if(it!=subkernelCallParams.end()){
                        subkernelCallParams.erase(it);
                    }
                }

                //Edit kernelcall function declaration to accomodate additional arguments added to kernel call expression.
                Rewrite.InsertTextAfterToken(kernelFunctionDecl->parameters()[kernelFunctionDecl->getNumParams()-1]->getLocEnd(),",int blocks,char *FL_Args");

                //Add preloop macro.
                Rewrite.InsertTextAfterToken(kernelCallBody->getLocStart(), "\n\tFL_T2_Preloop;\n");

                //Use the arguments left in map to compute required freelaunch variables.
                std::string subkernelCallStr="\n\tint FL_lc = atomicAdd(&FL_count,1);";
                subkernelCallStr+="\n\tFL_childKernelArgSz = sizeof(int)";
                for(std::map<std::string,std::string>::const_iterator it = subkernelCallParams.begin();it != subkernelCallParams.end(); ++it)
                                       {

                            subkernelCallStr+="+sizeof("+it->second+")";
                       }
                subkernelCallStr+=";";


                subkernelCallStr+="\n\tchar * _tmp_p = (char *) ((&FL_Args[0])+FL_lc*FL_childKernelArgSz);";
                subkernelCallStr+="\n\tint _tmp_childGridSize = "+subKernelCallpreArgs[0]+";\n\tmemcpy((void*)_tmp_p, (void*) &_tmp_childGridSize, sizeof(int));\n\t_tmp_p+=sizeof(int);\n\tFL_childBlockSize = "+subKernelCallpreArgs[1]+";";

                for(std::map<std::string,std::string>::const_iterator it = subkernelCallParams.begin();it != subkernelCallParams.end(); ++it)
                               {
                            std::map<std::string,std::string>::const_iterator itcopy=it;
                            ++itcopy;

                        if(itcopy==subkernelCallParams.end())
                            subkernelCallStr+="\n\tmemcpy((void*)_tmp_p, (void*) &"+it->first+", sizeof("+it->second+"));";

                        else if(it->second.compare("_Bool") == 0)
                            subkernelCallStr+="\n\tmemcpy((void*)_tmp_p, (void*) &"+it->first+", sizeof("+it->second+"));\n_tmp_p+=sizeof("+it->second+");";

                        else
                            subkernelCallStr+="\n\tmemcpy((void*)_tmp_p, (void*) &"+it->first+", sizeof("+it->second+"));\n_tmp_p+=sizeof(unsigned int);";
                       }

                subkernelCallStr+="\n\tFL_check = 0;\n\tgoto P;\n\t\tC: __threadfence()";

                //Replace the subkernel call expression with above assembled expression.
                Rewrite.ReplaceText(SourceRange(subKernelCall->getLocStart(),subKernelCall->getLocEnd()),subkernelCallStr);

                //Insert postLoop macro.
                Rewrite.InsertText(kernelCallBody->getLocEnd(), "\n\tFL_T2_Postloop;\n", true, true);

                //Post loop computations
                std::string postloopstr="\n\tchar * _tmp_p = (char*)((&FL_Args[0])+ckernelSeqNum*FL_childKernelArgSz);\n\tint kernelSz;\n\tmemcpy((void*)&kernelSz, (void*)_tmp_p, sizeof(int));\n\t_tmp_p+=sizeof(int);";

                for(std::map<std::string,std::string>::const_iterator it = subkernelCallParams.begin();it != subkernelCallParams.end(); ++it)
                    {
                        std::map<std::string,std::string>::const_iterator itcopy=it;
                        ++itcopy;

                        if(itcopy==subkernelCallParams.end())
                            postloopstr+="\n\t"+it->second+" "+it->first+";\n\tmemcpy((void*)&"+it->first+", (void*)_tmp_p, sizeof("+it->second+"));\n";

                        else if(it->second.compare("_Bool") == 0)
                            postloopstr+="\n\t"+it->second+" "+it->first+";\n\tmemcpy((void*)&"+it->first+", (void*)_tmp_p, sizeof("+it->second+"));\n\t_tmp_p+=sizeof("+it->second+");\n";

                        else
                            postloopstr+="\n\t"+it->second+" "+it->first+";\n\tmemcpy((void*)&"+it->first+", (void*)_tmp_p, sizeof("+it->second+"));\n\t_tmp_p+=sizeof(unsigned int);\n";
                    }

                Rewrite.InsertText(kernelCallBody->getLocEnd(),postloopstr+"\nfor(int k=0; k< kernelSz; k++){", true, true);

                //Logic to copy child kernel function body into kernel function body
                llvm::StringRef childbodytext = Lexer::getSourceText(CharSourceRange::getTokenRange(SourceRange(childCallBody->getLocStart(),childCallBody->getLocEnd())),SM, LangOpts);
                //get rid of enclosing {}
                std::string strTrimmed= childbodytext.str().substr(1,childbodytext.str().length()-2);
                //replace all return statements with continue.
                replaceAll(strTrimmed,"return","continue");
                //Replace the child thread id calculation expression.
                replaceAll(strTrimmed,"threadIdx.x + blockIdx.x*blockDim.x","k * FL_childBlockSize + threadIdx.x%FL_childBlockSize");
                replaceAll(strTrimmed,"threadIdx.x+ blockIdx.x*blockDim.x","k * FL_childBlockSize + threadIdx.x%FL_childBlockSize");
                replaceAll(strTrimmed,"threadIdx.x +blockIdx.x*blockDim.x","k * FL_childBlockSize + threadIdx.x%FL_childBlockSize");
                replaceAll(strTrimmed,"threadIdx.x+blockIdx.x*blockDim.x","k * FL_childBlockSize + threadIdx.x%FL_childBlockSize");
                Rewrite.InsertText(kernelCallBody->getLocEnd(),strTrimmed, true, true);
                //Insert postchildlog macro.
                Rewrite.InsertText(kernelCallBody->getLocEnd(),"\n\tFL_postChildLog\n", true, true);
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

    private:
              Rewriter &Rewrite;
};



class cudaSubKernelReturnCallHandler : public MatchFinder::MatchCallback {
        public:
                     cudaSubKernelReturnCallHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

                      virtual void run(const MatchFinder::MatchResult &Result) {
                     const ReturnStmt *returnstmt= Result.Nodes.getNodeAs<ReturnStmt>("returnstmt");
                 //replace all return statements in kernel function body with continue.
                 Rewrite.ReplaceText(SourceRange(returnstmt->getLocStart(),returnstmt->getLocEnd()),"continue");
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
                 if(blockoperand=="blockIdx.x"){
                         Rewrite.InsertText(blockstmtlhs->getLocStart(),"(",true,true);
                     Rewrite.InsertTextAfterToken(blockstmtlhs->getLocEnd(),"+FL_y)");

                 }
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

                Matcher.addMatcher(cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr().bind("subKernelCall"))).bind("cudaKernelCallDecl"))).bind("cudaKernelCall"), &HandlerForCudaSubKernCall);
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
