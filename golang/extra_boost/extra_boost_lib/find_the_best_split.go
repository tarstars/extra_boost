package extra_boost_lib

import (
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
)

//BestSplit contains results of the split selection algorithm.
type BestSplit struct {
	bestValue, currentValue          float64
	featureIndex, orderIndex         int
	threshold                        float64
	deltaUp, deltaDown, deltaCurrent *mat.Dense
	validSplit                       bool
}

//IterateSplits iterates through splits, incrementally updates hessian and gradient and
//calculates optimal weights difference and loss difference.
func IterateSplits(
	indRange IntIterable,
	em *EMatrix,
	featuresAs []int,
	bias *mat.Dense,
	currentLoss SplitLoss,
	d int,
	accumGrad *mat.Dense,
	rawHessian *tensor.Dense,
	accumHess *mat.Dense,
	parLambda float64,
	normHess *mat.Dense,
	inverseHess *mat.Dense,
	weight *mat.Dense,
	deltaLoss *mat.Dense,
	deltasLossCollection *mat.Dense,
	weightsCollection *mat.Dense,
) {
	for indRange.HasNext() {
		ind := indRange.GetNext()
		targetVal := em.target.At(featuresAs[ind], 0)
		biasVal := bias.At(featuresAs[ind], 0)
		der1 := currentLoss.lossDer1(targetVal, biasVal)
		der2 := currentLoss.lossDer2(targetVal, biasVal)

		for cp := 0; cp < d; cp++ {
			elemGrad := em.featuresExtra.At(featuresAs[ind], cp)
			accumGrad.Set(cp, 0, accumGrad.At(cp, 0)+der1*elemGrad)

			for cq := 0; cq < d; cq++ {
				element, err := rawHessian.At(featuresAs[ind], cp, cq)
				handleError(err)
				accumHess.Set(cp, cq, der2*element.(float64)+accumHess.At(cp, cq))
				diagEye := 0.0
				if cp == cq {
					diagEye = parLambda
				}
				normHess.Set(cp, cq, accumHess.At(cp, cq)+diagEye)
			}
		}

		handleError(inverseHess.Inverse(normHess))
		weight.Mul(inverseHess, accumGrad)
		deltaLoss.Mul(weight.T(), accumGrad)
		deltasLossCollection.Set(ind, 0, deltaLoss.At(0, 0))
		weightsCollection.SetRow(ind, weight.RawMatrix().Data)
	}
}

//flushIntermediate flushes the gradient and the hessian
func flushIntermediate(d int, accumGrad *mat.Dense, accumHess *mat.Dense) {
	for zeroIndP := 0; zeroIndP < d; zeroIndP++ {
		accumGrad.Set(zeroIndP, 0, 0)
		for zeroIndQ := 0; zeroIndQ < d; zeroIndQ++ {
			accumHess.Set(zeroIndP, zeroIndQ, 0)
		}
	}
}

//selectTheBestSplit scans through different splits and selects the best one
func selectTheBestSplit(em EMatrix, featuresAs []int, bestSplit *BestSplit, h, q, d int, deltasLossUp, deltasLossDown, weightsUp, weightsDown *mat.Dense) {
	firstIter := true

	bestSplit.featureIndex = q
	for hInd := 0; hInd < h-1; hInd++ {
		currentLossValue := -0.5 * (deltasLossUp.At(hInd, 0) + deltasLossDown.At(hInd+1, 0))
		for ; hInd < h-1 && em.featuresInter.At(featuresAs[hInd], q) == em.featuresInter.At(featuresAs[hInd+1], q); hInd++ {
		}
		if hInd < h-1 && (firstIter || bestSplit.bestValue > currentLossValue) {
			firstIter = false
			bestSplit.bestValue = currentLossValue
			for qInd := 0; qInd < d; qInd++ {
				bestSplit.deltaUp.Set(qInd, 0, weightsUp.At(hInd, qInd))
				bestSplit.deltaDown.Set(qInd, 0, weightsDown.At(hInd+1, qInd))
			}
			bestSplit.threshold = (em.featuresInter.At(featuresAs[hInd], q) + em.featuresInter.At(featuresAs[hInd+1], q)) / 2
			bestSplit.orderIndex = hInd
		}
	}
	for ind := 0; ind < d; ind++ {
		bestSplit.deltaCurrent.Set(ind, 0, weightsUp.At(h-1, ind))
	}
	bestSplit.validSplit = !firstIter
}

//scanForSplit allocates memory, performs argsort of selected feature column,
//iterates through splits upside down and downside up and selects the best split
//in the current column.
func scanForSplit(
	em EMatrix,
	h, d, q int,
	bias *mat.Dense,
	currentLoss SplitLoss,
	parLambda float64,
	rawHessian *tensor.Dense,
) (bestSplit BestSplit) {
	featuresAs := columnArgsort(em.featuresInter.ColView(q))

	accumHess := mat.NewDense(d, d, nil)
	normHess := mat.NewDense(d, d, nil)
	inverseHess := mat.NewDense(d, d, nil)

	accumGrad := mat.NewDense(d, 1, nil)
	weight := mat.NewDense(d, 1, nil)
	deltaLoss := mat.NewDense(1, 1, nil)

	deltasLossUp := mat.NewDense(h, 1, nil)
	weightsUp := mat.NewDense(h, d, nil)
	deltasLossDown := mat.NewDense(h, 1, nil)
	weightsDown := mat.NewDense(h, d, nil)

	bestSplit.deltaUp = mat.NewDense(d, 1, nil)
	bestSplit.deltaDown = mat.NewDense(d, 1, nil)
	bestSplit.deltaCurrent = mat.NewDense(d, 1, nil)

	IterateSplits(NewRange(0, h, 1), &em, featuresAs,
		bias, currentLoss, d, accumGrad, rawHessian, accumHess, parLambda,
		normHess, inverseHess, weight, deltaLoss, deltasLossUp, weightsUp)

	flushIntermediate(d, accumGrad, accumHess)

	IterateSplits(NewRange(h-1, -1, -1), &em, featuresAs,
		bias, currentLoss, d, accumGrad, rawHessian, accumHess, parLambda,
		normHess, inverseHess, weight, deltaLoss, deltasLossDown, weightsDown)

	selectTheBestSplit(em, featuresAs, &bestSplit, h, q, d, deltasLossUp, deltasLossDown, weightsUp, weightsDown)

	return
}

//allocateArrays allocates the raw hessian array.
func (em EMatrix) allocateArrays() (rawHessian *tensor.Dense) {
	h, _ := em.featuresInter.Dims()
	_, d := em.featuresExtra.Dims()

	rawHessian = tensor.New(tensor.WithShape(h, d, d), tensor.Of(tensor.Float64))
	for p := 0; p < h; p++ {
		for q := 0; q < d; q++ {
			for r := 0; r < d; r++ {
				handleError(rawHessian.SetAt(em.featuresExtra.At(p, q)*em.featuresExtra.At(p, r), p, q, r))
			}
		}
	}
	return
}
