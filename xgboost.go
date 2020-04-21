package xgb

import (
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/xgb/capi"
	"runtime"
	"unsafe"
)

func LibVersion() fu.VersionType {
	return capi.LibVersion
}

type xgbinstance struct {
	handle   unsafe.Pointer
	features []string // names of features
	predicts string   // name of predicted value
}

func (x *xgbinstance) Close() (_ error) {
	runtime.SetFinalizer(x, nil)
	capi.Close(x.handle)
	x.handle = nil
	return
}
