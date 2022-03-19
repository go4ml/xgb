[![CircleCI](https://circleci.com/gh/go-ml-dev/xgb.svg?style=svg)](https://circleci.com/gh/go-ml-dev/xgb)
[![Maintainability](https://api.codeclimate.com/v1/badges/24971185cab6e6a3fae2/maintainability)](https://codeclimate.com/github/go-ml-dev/xgb/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/24971185cab6e6a3fae2/test_coverage)](https://codeclimate.com/github/go-ml-dev/xgb/test_coverage)
[![Go Report Card](https://goreportcard.com/badge/github.com/go-ml-dev/xgb)](https://goreportcard.com/report/github.com/go-ml-dev/xgb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


```golang
import (
	"fmt"
	"go4ml.xyz/base/model"
	"go4ml.xyz/dataset/mnist"
	"go4ml.xyz/iokit"
	"go4ml.xyz/xgb"
	"gotest.tools/assert"
	"testing"
)

func Test_minstXgb(t *testing.T) {
	modelFile := iokit.File(model.Path("mnist_test_xgb.zip"))
	report := xgb.Model{
		Algorithm:    xgb.TreeBoost,
		Function:     xgb.Softmax,
		LearningRate: 0.54,
		MaxDepth:     7,
		Extra:        map[string]interface{}{"tree_method": "hist"},
	}.Feed(model.Dataset{
		Source:   mnist.Data.RandomFlag(model.TestCol, 42, 0.1),
		Features: mnist.Features,
	}).LuckyTrain(model.Training{
		Iterations: 30,
		ModelFile:  modelFile,
		Metrics:    model.Classification{Accuracy: 0.96},
		Score:      model.AccuracyScore,
	})

	fmt.Println(report.TheBest, report.Score)
	fmt.Println(report.History.Round(5))
	assert.Assert(t, model.Accuracy(report.Test) >= 0.96)

	pred := xgb.LuckyObjectify(modelFile)
	lr := model.LuckyEvaluate(mnist.T10k, model.LabelCol, pred, 32, model.Classification{})
	fmt.Println(lr.Round(5))
	assert.Assert(t, model.Accuracy(lr) >= 0.96)
}
```
