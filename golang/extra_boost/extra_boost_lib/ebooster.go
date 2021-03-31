package extra_boost_lib

import (
	"gonum.org/v1/gonum/mat"
	"log"
)

//PredictOperator infers operator that converts extra features into a prediction.
func (oneTree OneTree) PredictOperator(featuresInter *mat.Dense) (prediction *mat.Dense) {
	h, _ := featuresInter.Dims()
	prediction = mat.NewDense(h, oneTree.d, nil)

	for p := 0; p < h; p++ {
		ind := 0
		for oneTree.TreeNodes[ind].LeafIndex == -1 {
			if featuresInter.At(p, oneTree.TreeNodes[ind].FeatureNumber) < oneTree.TreeNodes[ind].Threshold {
				ind = oneTree.TreeNodes[ind].LeftIndex
			} else {
				ind = oneTree.TreeNodes[ind].RightIndex
			}
		}
		prediction.SetRow(p, oneTree.LeafNodes[oneTree.TreeNodes[ind].LeafIndex].Prediction)
	}

	return
}

// PredictValue infers values of a model by inferring an operator and applying it to the Extra data.
func (oneTree OneTree) PredictValue(featuresInter, featuresExtra *mat.Dense) (prediction *mat.Dense) {
	operator := oneTree.PredictOperator(featuresInter)
	h, _ := featuresInter.Dims()
	prediction = mat.NewDense(h, 1, nil)
	for p := 0; p < h; p++ {
		s := 0.0
		for q := 0; q < oneTree.d; q++ {
			s += operator.At(p, q) * featuresExtra.At(p, q)
		}
		prediction.Set(p, 0, s)
	}
	return
}

//EBooster is the model class.
type EBooster struct {
	Trees []OneTree
}

//NewEBooster creates a new model.
func NewEBooster(ematrix EMatrix, nStages int, regLambda float64, maxDepth int, learningRate float64, lossKind SplitLoss, printMessages []EMatrix) (ebooster *EBooster) {
	ebooster = &EBooster{make([]OneTree, 0)}
	h, _ := ematrix.featuresInter.Dims()
	bias := mat.NewDense(h, 1, nil)
	for stage := 0; stage < nStages; stage++ {
		log.Printf("Tree number %d\n", stage+1)
		tree := NewTree(ematrix, bias, regLambda, maxDepth, learningRate, lossKind)
		ebooster.Trees = append(ebooster.Trees, tree)
		deltaB := tree.PredictValue(ematrix.featuresInter, ematrix.featuresExtra)
		bias.Add(bias, deltaB)
		for _, currentEmatrix := range printMessages {
			currentEmatrix.Message(*ebooster)
		}
	}
	return
}

//PredictValue infers values of the target. It requires both sets of features - interpolating and extrapolating.
func (ebooster EBooster) PredictValue(featuresInter, featuresExtra *mat.Dense) (prediction *mat.Dense) {
	prediction = ebooster.Trees[0].PredictValue(featuresInter, featuresExtra)
	n := len(ebooster.Trees)
	for treeInd := 1; treeInd < n; treeInd++ {
		deltaPrediction := ebooster.Trees[treeInd].PredictValue(featuresInter, featuresExtra)
		prediction.Add(prediction, deltaPrediction)
	}
	return
}
