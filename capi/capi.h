
#include<stdint.h>

typedef void *DMatrixHandle;
typedef void *BoosterHandle;
typedef uint64_t bst_ulong;

void XGBoostVersion(int* major, int* minor, int* patch);
const char *XGBGetLastError(void);
int XGDMatrixCreateFromFile(const char *fname,int silent,DMatrixHandle *out);
int XGDMatrixCreateFromDT(void** data,
    const char ** feature_stypes,
    bst_ulong nrow,
    bst_ulong ncol,
    DMatrixHandle* out,
    int nthread);
int XGDMatrixCreateFromMat(const float *data,
    bst_ulong nrow,
    bst_ulong ncol,
    float missing,
    DMatrixHandle *out);
int XGDMatrixFree(DMatrixHandle handle);
int XGDMatrixSaveBinary(DMatrixHandle handle,const char *fname, int silent);
int XGDMatrixNumRow(DMatrixHandle handle,bst_ulong *out);
int XGDMatrixNumCol(DMatrixHandle handle,bst_ulong *out);
int XGBoosterCreate(const DMatrixHandle dmats[],bst_ulong len,BoosterHandle *out);
int XGBoosterFree(BoosterHandle handle);
int XGBoosterSetParam(BoosterHandle handle,const char *name,const char *value);
int XGBoosterLoadModel(BoosterHandle handle,const char *fname);
int XGBoosterSaveModel(BoosterHandle handle,const char *fname);
int XGBoosterLoadModelFromBuffer(BoosterHandle handle,const void *buf,bst_ulong len);
int XGBoosterGetModelRaw(BoosterHandle handle, bst_ulong *out_len, /* const char* */ void *out_dptr);
int XGBoosterLoadJsonConfig(BoosterHandle handle,char const *json_parameters);
int XGBoosterSaveJsonConfig(BoosterHandle handle, bst_ulong *out_len,/* char const* */ void *out_str);
int XGBoosterGetAttr(BoosterHandle handle,const char* key,const char** out,int *success);
int XGBoosterSetAttr(BoosterHandle handle,const char* key,const char* value);
int XGBoosterGetAttrNames(BoosterHandle handle,bst_ulong* out_len,const char*** out);
int XGBoosterBoostOneIter(BoosterHandle handle,
    DMatrixHandle dtrain,
    float *grad,
    float *hess,
    bst_ulong len);
int XGBoosterUpdateOneIter(BoosterHandle handle,int iter,DMatrixHandle dtrain);
int XGBoosterEvalOneIter(BoosterHandle handle,
    int iter,
    DMatrixHandle dmats[],
    const char *evnames[],
    bst_ulong len,
    const char **out_result);
int XGBoosterPredict(BoosterHandle handle,
    DMatrixHandle dmat,
    int option_mask,
    unsigned ntree_limit,
    int training,
    bst_ulong *out_len,
    /* const float* */ void *out_result);
int XGDMatrixSetFloatInfo(DMatrixHandle handle,
    const char *field,
    const float *array,
    bst_ulong len);
int XGDMatrixSetUIntInfo(DMatrixHandle handle,
    const char *field,
    const unsigned *array,
    bst_ulong len);
int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
    const char *field,
    bst_ulong* out_len,
    const float **out_dptr);
int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
    const char *field,
    bst_ulong* out_len,
    const unsigned **out_dptr);
int XGBoosterSerializeToBuffer(BoosterHandle handle,
    bst_ulong *out_len,
    const char **out_dptr);
int XGBoosterUnserializeFromBuffer(BoosterHandle handle,
    const void *buf,
    bst_ulong len);
int XGBoosterDumpModelEx(BoosterHandle handle,
    const char *fmap,
    int with_stats,
    const char *format,
    bst_ulong *out_len,
    /*const char **/ void *out_dump_array);

#define DEFINE_JUMPER(x) \
        void *_godl_##x = (void*)0; \
        __asm__(".global "#x"\n\t"#x":\n\tmovq _godl_"#x"(%rip),%rax\n\tjmp *%rax\n")

DEFINE_JUMPER(XGBoostVersion);
DEFINE_JUMPER(XGBGetLastError);
DEFINE_JUMPER(XGDMatrixCreateFromFile);
DEFINE_JUMPER(XGDMatrixCreateFromDT);
DEFINE_JUMPER(XGDMatrixCreateFromMat);
DEFINE_JUMPER(XGDMatrixFree);
DEFINE_JUMPER(XGDMatrixSaveBinary);
DEFINE_JUMPER(XGDMatrixNumRow);
DEFINE_JUMPER(XGDMatrixNumCol);
DEFINE_JUMPER(XGBoosterCreate);
DEFINE_JUMPER(XGBoosterFree);
DEFINE_JUMPER(XGBoosterSetParam);
DEFINE_JUMPER(XGBoosterLoadModel);
DEFINE_JUMPER(XGBoosterSaveModel);
DEFINE_JUMPER(XGBoosterLoadModelFromBuffer);
DEFINE_JUMPER(XGBoosterGetModelRaw);
DEFINE_JUMPER(XGBoosterSaveJsonConfig);
DEFINE_JUMPER(XGBoosterLoadJsonConfig);
DEFINE_JUMPER(XGBoosterGetAttr);
DEFINE_JUMPER(XGBoosterSetAttr);
DEFINE_JUMPER(XGBoosterGetAttrNames);
DEFINE_JUMPER(XGBoosterBoostOneIter);
DEFINE_JUMPER(XGBoosterUpdateOneIter);
DEFINE_JUMPER(XGBoosterEvalOneIter);
DEFINE_JUMPER(XGBoosterPredict);
DEFINE_JUMPER(XGDMatrixSetFloatInfo);
DEFINE_JUMPER(XGDMatrixSetUIntInfo);
DEFINE_JUMPER(XGDMatrixGetFloatInfo);
DEFINE_JUMPER(XGDMatrixGetUIntInfo);
DEFINE_JUMPER(XGBoosterDumpModelEx);
