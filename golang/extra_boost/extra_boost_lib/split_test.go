package extra_boost_lib

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func CreateTestEMatrix() (EMatrix, int) {
	h := 10
	extraW := 3

	AllWeights := []*mat.Dense{mat.NewDense(extraW, 1, []float64{7, 11, 13}), mat.NewDense(extraW, 1, []float64{1979, 18, 25})}
	nWeights := len(AllWeights)

	rawTarget := make([]float64, nWeights*h)
	target := mat.NewDense(h*nWeights, 1, rawTarget)

	parabolaArray := make([]float64, nWeights*h*extraW)

	rangeArray := make([]float64, h*nWeights)
	featuresInter := mat.NewDense(nWeights*h, 1, rangeArray)

	for weightIndex, currentWeight := range AllWeights {
		for ind := 0; ind < h; ind++ {
			rangeArray[weightIndex*h+ind] = float64(weightIndex*h + ind + 1)
		}

		timeAperture := 1.0
		for p := 0; p < h; p++ {
			t := float64(p) * timeAperture / (float64(h) - 1.0)

			parabolaArray[weightIndex*h*extraW+extraW*p+0] = 1.0
			parabolaArray[weightIndex*h*extraW+extraW*p+1] = t
			parabolaArray[weightIndex*h*extraW+extraW*p+2] = t * t

			currentTarget := 0.0
			for pp := 0; pp < extraW; pp++ {
				currentTarget += parabolaArray[weightIndex*h*extraW+extraW*p+pp] * currentWeight.At(pp, 0)
			}
			target.Set(h*weightIndex+p, 0, currentTarget)
		}
	}
	featuresExtra := mat.NewDense(nWeights*h, extraW, parabolaArray)

	//fmt.Println("featuresInter")
	//fmt.Printf("%.4g\n", mat.Formatted(featuresInter))
	//fmt.Println("featuresExtra")
	//fmt.Printf("%.4g\n", mat.Formatted(featuresExtra))
	//fmt.Println("target")
	//fmt.Printf("%.4g\n", mat.Formatted(target))

	return EMatrix{featuresInter, featuresExtra, target}, nWeights
}

func TestScanForSplit(t *testing.T) {
	testEMatrix, nWeights := CreateTestEMatrix()

	h, _, d := testEMatrix.validatedDimensions()
	rawHessian := testEMatrix.allocateArrays()

	bias := mat.NewDense(nWeights*h, 1, nil)

	bestSplit := scanForSplit(testEMatrix, h, d, 0, bias, MseLoss{}, 1e-6, rawHessian)

	fmt.Println("delta up:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaUp))

	fmt.Println("delta down:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaDown))

	fmt.Println("delta current:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaCurrent))
}
