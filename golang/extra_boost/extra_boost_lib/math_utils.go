package extra_boost_lib

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

//handleError handles an error in the simplest possible way: it panics
func handleError(err error) {
	if err != nil {
		panic(err)
	}
}

//sigmoid64 calculates the sigmoid function
func sigmoid64(x float64) float64 {
	if x < -30 {
		return 0.0
	}
	if x > 30 {
		return 1.0
	}
	return 1.0 / (1 + math.Exp(-x))
}

//Height calculates the height of a matrix m.
func Height(m *mat.Dense) int {
	h, _ := m.Dims()
	return h
}

//Rmse calculates rooted mean squared difference.
func Rmse(current, other *mat.Dense) float64 {
	if Height(current) != Height(other) {
		panic("Shapes of matrix in Rmse differ")
	}

	n := Height(current)
	s := 0.0
	for ind := 0; ind < n; ind++ {
		d := current.At(ind, 0) - other.At(ind, 0)
		s += d * d
	}

	return math.Sqrt(s / float64(n))
}

//Logloss calculates the logloss function. It is possible to apply the sigmoid function to the score.
func Logloss(target, score *mat.Dense, applySigmoid bool) float64 {
	if Height(target) != Height(score) {
		panic("Shapes of matrix in Rmse differ")
	}

	n := Height(target)
	s := 0.0
	for ind := 0; ind < n; ind++ {
		scoreElement := score.At(ind, 0)
		if applySigmoid {
			scoreElement = sigmoid64(scoreElement)
		}
		targetElement := target.At(ind, 0)
		s += targetElement*math.Log(scoreElement) + (1.0-targetElement)*math.Log(1.0-scoreElement)
	}

	return s
}
