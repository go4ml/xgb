package xgb

import (
	"go4ml.xyz/xgb/capi"
)

type capiparam interface{ pair() (string, string) }

func (x xgbinstance) setparam(par capiparam) {
	n, v := par.pair()
	capi.SetParam(x.handle, n, v)
}

type booster string

func (b booster) pair() (string, string) { return "booster", string(b) }

const TreeBoost = booster("gbtree")
const LinearBoost = booster("gblinear")
const DartBoost = booster("dart")

type objective string

func (o objective) pair() (string, string) { return "objective", string(o) }

const Linear = objective("reg:linear")
const SquareLinear = objective("reg:squarederror")
const Logistic = objective("reg:logistic")
const SqureLogistic = objective("reg:squaredlogerror")
const Tweedie = objective("reg:tweedie")
const Binary = objective("binary:logistic")
const RawBinary = objective("binary:logitraw")
const HingeBinary = objective("binary:hinge")

// gamma regression with log-link. Output is a mean of gamma distribution.
// It might be useful, e.g., for modeling insurance claims severity,
// or for any outcome that might be gamma-distributed.
const GammaRegress = objective("reg:gamma")

// set XGBoost to do multiclass classification using the softmax objective,
// you also need to set num_class(number of classes)
const Softmax = objective("multi:softmax")

// same as softmax, but output a vector of ndata * nclass,
// which can be further reshaped to ndata * nclass matrix.
// The result contains predicted probability of each data point belonging to each class.
const Softprob = objective("multi:softprob")

type Param struct{ Name, Value string }

func (sp Param) pair() (string, string) { return sp.Name, sp.Value }
