package extra_boost_lib

//SplitLoss interface is the interface for a loss function. It provides the first and the second derivatives.
type SplitLoss interface {
	lossDer1(float64, float64) float64
	lossDer2(float64, float64) float64
}

//Logloss struct.
type LogLoss struct{}

//lossDer1 calculates the first derivative for the logloss
func (LogLoss) lossDer1(target, bias float64) float64 {
	return -target*sigmoid64(-bias) + (1.0-target)*sigmoid64(bias)
}

//lossDer2 calculates the second derivative for the logloss
func (LogLoss) lossDer2(_, bias float64) float64 {
	return sigmoid64(-bias) * sigmoid64(bias)
}

//MseLoss contains methods for mean squared error loss
type MseLoss struct{}

//lossDer1 the first derivation of the loss
func (MseLoss) lossDer1(target, bias float64) float64 {
	return target - bias
}

//lossDer2 the second derivation of the loss
func (MseLoss) lossDer2(_, _ float64) float64 {
	return 1.0
}
