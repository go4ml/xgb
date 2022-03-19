package xgb

import (
	"fmt"
	"go4ml.xyz/base/fu"
	"go4ml.xyz/base/model"
	"go4ml.xyz/base/tables"
	"go4ml.xyz/xgb/capi"
	"go4ml.xyz/zorros"
	"unsafe"
)

func train(e Model, dataset model.Dataset, w model.Workout) (report *model.Report, err error) {

	t, err := dataset.Source.Collect()
	if err != nil {
		return
	}

	Test := fu.Fnzs(dataset.Test, model.TestCol)
	if fu.IndexOf(Test, t.Names()) < 0 {
		err = zorros.Errorf("dataset does not have column `%v`", Test)
		return
	}

	Label := fu.Fnzs(dataset.Label, model.LabelCol)
	if fu.IndexOf(Label, t.Names()) < 0 {
		err = zorros.Errorf("dataset does not have column `%v`", Label)
		return
	}

	features := t.OnlyNames(dataset.Features...)
	test, train, err := t.MatrixWithLabelIf(features, Label, Test, true)
	if err != nil {
		return
	}

	m := matrix32(train)
	defer m.Free()
	m2 := matrix32(test)
	defer m2.Free()

	predicts := fu.Fnzs(e.Predicted, model.PredictedCol)

	xgb := &xgbinstance{capi.Create2(m.handle, m2.handle), features, predicts}
	defer xgb.Close()

	if e.Algorithm != booster("") {
		xgb.setparam(e.Algorithm)
	}

	if e.Function != objective("") {
		xgb.setparam(e.Function)
	}

	if e.LearningRate != 0 {
		capi.SetParam(xgb.handle, "eta", fmt.Sprint(e.LearningRate))
	}

	if e.MaxDepth != 0 {
		capi.SetParam(xgb.handle, "max_depth", fmt.Sprint(e.MaxDepth))
	}

	capi.SetParam(xgb.handle, "num_feature", fmt.Sprint(len(features)))
	if e.Function == Softprob || e.Function == Softmax {
		x := int(fu.Maxr(fu.Maxr(0, train.Labels...), test.Labels...))
		if x < 0 {
			panic(zorros.Errorf("labels don't contain enough classes or label values is incorrect"))
		}
		capi.SetParam(xgb.handle, "num_class", fmt.Sprint(x+1))
	}

	testLabels := test.AsLabelColumn()
	trainLabels := train.AsLabelColumn()

	for done := false; w != nil && !done; w = w.Next() {
		capi.Update(xgb.handle, w.Iteration(), m.handle)
		m0, _ := xgb.metrics(m.handle, trainLabels, w.TrainMetrics())
		m1, d := xgb.metrics(m2.handle, testLabels, w.TestMetrics())
		if report, done, err = w.Complete(model.MemorizeMap{"model": mnemosyne{xgb}}, m0, m1, d); err != nil {
			return nil, zorros.Wrapf(err, "tailed to complete model: %s", err.Error())
		}
	}

	return
}

func (xgb *xgbinstance) metrics(m unsafe.Pointer, label *tables.Column, mu model.MetricsUpdater) (fu.Struct, bool) {
	y := capi.Predict(xgb.handle, m, 0)
	pred := tables.Matrix{
		Features:    y,
		Labels:      nil,
		Width:       len(y) / label.Len(),
		Length:      label.Len(),
		LabelsWidth: 0,
	}
	model.BatchUpdateMetrics(pred.AsColumn(), label, mu)
	return mu.Complete()
}
