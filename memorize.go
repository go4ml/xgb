package xgb

import (
	"bufio"
	"encoding/json"
	"go4ml.xyz/base/model"
	"go4ml.xyz/xgb/capi"
	"io"
)

type mnemosyne struct{ *xgbinstance }

func (x mnemosyne) Memorize(c *model.CollectionWriter) (err error) {
	if err = c.Add("info.json", func(wr io.Writer) error {
		en := json.NewEncoder(wr)
		return en.Encode(map[string]interface{}{
			"features": x.features,
			"predicts": x.predicts,
		})
	}); err != nil {
		return
	}
	if err = c.Add("config.json", func(wr io.Writer) error {
		_, err := wr.Write(capi.JsonConfig(x.handle))
		return err
	}); err != nil {
		return
	}
	if err = c.AddLzma2("model.bin.xz", func(wr io.Writer) error {
		_, err := wr.Write(capi.GetModel(x.handle))
		return err
	}); err != nil {
		return
	}
	if err = c.AddLzma2("dump.txt.xz", func(wr io.Writer) error {
		w := bufio.NewWriter(wr)
		for _, s := range capi.DumpModel(x.handle) {
			_, err := w.WriteString(s)
			if err != nil {
				return err
			}
		}
		return w.Flush()
	}); err != nil {
		return
	}
	return
}
