package main

import (
	ebl "example.com/extra_boost_lib"
	"log"
	"path"
)

func main() {
	baseDir := "/home/tass/database/app_in_the_air/demand_predictions/current_data_set"
	log.Println("load train")
	ematrix_train := ebl.ReadEMatrix(
		path.Join(baseDir, "inter_train.npy"),
		path.Join(baseDir, "extra_train.npy"),
		path.Join(baseDir, "target_train.npy"),
	)

	log.Println("load test")
	ematrix_test := ebl.ReadEMatrix(
		path.Join(baseDir, "inter_test.npy"),
		path.Join(baseDir, "extra_test.npy"),
		path.Join(baseDir, "target_test.npy"),
	)

	clf := ebl.NewEBooster(ematrix_train, 20, 1e-6, 5, 0.3, ebl.MseLoss{}, []ebl.EMatrix{ematrix_train, ematrix_test})

	//fmt.Println(func() string { r, _ := json.MarshalIndent(clf, "", "  "); return string(r) }())
	_ = clf
}
