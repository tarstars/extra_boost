package extra_boost_lib

import (
	"github.com/sbinet/npyio"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
)

//EMatrix contains data for either MSE or LogLoss loss functions
type EMatrix struct {
	featuresInter *mat.Dense
	featuresExtra *mat.Dense
	target        *mat.Dense
}

//Message prints a message about the current state of the prediction on the current dataset
func (ematrix EMatrix) Message(ebooster EBooster) {
	prediction := ebooster.PredictValue(ematrix.featuresInter, ematrix.featuresExtra)
	log.Print("RMSE =", Rmse(ematrix.target, prediction))
}

//ReadEMatrix reads three components of a data set and unites them into one EMatrix object
func ReadEMatrix(fileNameInter, fileNameExtra, fileNameTarget string) (em EMatrix) {
	log.Print("\ttry to load inter")
	em.featuresInter = readNpy(fileNameInter)
	log.Print("\ttry to load extra")
	em.featuresExtra = readNpy(fileNameExtra)
	log.Print("\ttry to load target")
	em.target = readNpy(fileNameTarget)

	return
}

//readNpy reads the content of npy file
func readNpy(fileName string) *mat.Dense {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { handleError(f.Close()) }()

	r, err := npyio.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}

	shape := r.Header.Descr.Shape
	raw := make([]float64, shape[0]*shape[1])

	err = r.Read(&raw)
	if err != nil {
		log.Fatal(err)
	}

	return mat.NewDense(shape[0], shape[1], raw)
}

//Split splits data of receiver by the BestSplit criterion
func (em EMatrix) Split(bias *mat.Dense, split BestSplit) (leftEmatrix, rightEmatrix EMatrix, leftBias, rightBias *mat.Dense) {
	h, w := em.featuresInter.Dims()
	_, extraW := em.featuresExtra.Dims()
	leftCount, rightCount := 0, 0

	for p := 0; p < h; p++ {
		if em.featuresInter.At(p, split.featureIndex) < split.threshold {
			leftCount++
		} else {
			rightCount++
		}
	}

	leftBias = mat.NewDense(leftCount, 1, nil)
	rightBias = mat.NewDense(rightCount, 1, nil)

	leftFeaturesInter := mat.NewDense(leftCount, w, nil)
	rightFeaturesInter := mat.NewDense(rightCount, w, nil)

	leftFeaturesExtra := mat.NewDense(leftCount, extraW, nil)
	rightFeaturesExtra := mat.NewDense(rightCount, extraW, nil)

	leftTarget := mat.NewDense(leftCount, 1, nil)
	rightTarget := mat.NewDense(rightCount, 1, nil)

	leftInd, rightInd := 0, 0

	for p := 0; p < h; p++ {
		if em.featuresInter.At(p, split.featureIndex) < split.threshold {
			leftBias.Set(leftInd, 0, bias.At(p, 0))
			for q := 0; q < w; q++ {
				leftFeaturesInter.Set(leftInd, q, em.featuresInter.At(p, q))
			}
			for q := 0; q < extraW; q++ {
				leftFeaturesExtra.Set(leftInd, q, em.featuresExtra.At(p, q))
			}
			leftTarget.Set(leftInd, 0, em.target.At(p, 0))
			leftInd++
		} else {
			rightBias.Set(rightInd, 0, bias.At(p, 0))
			for q := 0; q < w; q++ {
				rightFeaturesInter.Set(rightInd, q, em.featuresInter.At(p, q))
			}
			for q := 0; q < extraW; q++ {
				rightFeaturesExtra.Set(rightInd, q, em.featuresExtra.At(p, q))
			}
			rightTarget.Set(rightInd, 0, em.target.At(p, 0))
			rightInd++
		}
	}

	return EMatrix{leftFeaturesInter, leftFeaturesExtra, leftTarget}, EMatrix{rightFeaturesInter, rightFeaturesExtra, rightTarget}, leftBias, rightBias
}

//validateDimensions checks the consistency of dimensions in arrays from the current dataset
//and returns the height (the number of objects), the width (the number of features) and the depth
//(the number of extra features per record) of the current dataset.
func (em EMatrix) validatedDimensions() (h, w, d int) {
	h, w = em.featuresInter.Dims()
	extraH, d := em.featuresExtra.Dims()
	if extraH != h {
		log.Panicf("the extra height %d is not equal to the inter height %d", extraH, h)
	}
	targetH, targetW := em.target.Dims()
	if targetH != h {
		log.Panicf("the target height %d is not equal to the inter height %d", targetH, h)
	}
	if targetW != 1 {
		log.Panicf("the width of target should be 1 not %d", targetW)
	}
	return h, w, d
}
