package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	"log"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/sbinet/npyio"
)

//readNpz reads the content of non-compressed npy file
func readNpz(fileName string) *mat.Dense {
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

//EMatrix contains data for either MSE or LogLoss loss funtions
type EMatrix struct {
	featuresInter *mat.Dense
	featuresExtra *mat.Dense
	target        *mat.Dense
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

//ReadEMatrix reads three components of a data set and unites them into one EMatrix object
func ReadEMatrix(fileNameInter, fileNameExtra, fileNameTarget string) (em EMatrix) {
	em.featuresInter = readNpz(fileNameInter)
	em.featuresExtra = readNpz(fileNameExtra)
	em.target = readNpz(fileNameTarget)

	return
}

//VecSort is an interface for mat.Dense column
type VecSort interface {
	AtVec(int) float64
	Len() int
}

//argsort allows to implement an algorithm of column argsorting.
type argsort struct {
	c   VecSort
	ind []int
}

//Len returns the length of a column.
func (a argsort) Len() int {
	return a.c.Len()
}

//Less compares two elements.
func (a argsort) Less(i, j int) bool {
	return a.c.AtVec(a.ind[i]) < a.c.AtVec(a.ind[j])
}

//Swap swaps elements of inderect addressing array.
func (a argsort) Swap(i, j int) {
	a.ind[i], a.ind[j] = a.ind[j], a.ind[i]
}

//columnArgsort performs the argsort operation for a column of a matrix
func columnArgsort(c VecSort) []int {
	n := c.Len()
	result := make([]int, n)

	for ind := 0; ind < n; ind++ {
		result[ind] = ind
	}

	locArgsort := argsort{c, result}
	sort.Sort(locArgsort)

	return result
}

//handleError handles an error in the simplest possible way: it panics
func handleError(err error) {
	if err != nil {
		panic(err)
	}
}

//sigmoid64 calculates the sigmoid function
func sigmoid64(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

//SplitLoss interface is the interface for a loss function. It provides the first and the second derivatives.
type SplitLoss interface {
	lossDer1(float64, float64) float64
	lossDer2(float64, float64) float64
}

//Logloss struct.
type LogLoss struct{}

//lossDer1 the first derivative for the logloss
func (LogLoss) lossDer1(target, bias float64) float64 {
	return -target*sigmoid64(-bias) + (1.0-target)*sigmoid64(bias)
}

//lossDer2 the second derivative for the logloss
func (LogLoss) lossDer2(_, bias float64) float64 {
	return sigmoid64(-bias) * sigmoid64(bias)
}

//type MseLoss struct{}
//
//func (MseLoss) lossDer1(target, bias float64) float64 {
//	return target - bias
//}
//
//func (MseLoss) lossDer2(_, _ float64) float64 {
//	return 1.0
//}

//BestSplit contains results of the split selection algorithm.
type BestSplit struct {
	bestValue                float64
	featureIndex, orderIndex int
	threshold                float64
	deltaUp, deltaDown       *mat.Dense
}

//IntIterable is the interface for iteration over an collection of integers.
type IntIterable interface {
	HasNext() bool
	GetNext() int
}

//Range is an iterator over half inverval [begin, end) with the step step.
type Range struct {
	begin, end, step, pos int
}

//NewRange initializes a new iterator over a half interval.
func NewRange(start, end, step int) *Range {
	return &Range{start, end, step, start}
}

//GetNext returns the next element from the iterator and moves iterator to the next position.
func (r *Range) GetNext() int {
	val := r.pos
	r.pos += r.step
	return val
}

//HasNext checks whether there are more values in the iterator.
func (r *Range) HasNext() bool {
	if r.step > 0 {
		return r.pos < r.end
	}
	return r.pos > r.end
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
			accumGrad.Set(cp, 0, accumGrad.At(cp, 0)-der1*elemGrad)

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

//flushIntermediate flushes gradient and hessian
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
	for ind := 0; ind < h-1; ind++ {
		currentLossValue := -0.5 * (deltasLossUp.At(ind, 0) + deltasLossDown.At(ind+1, 0))
		if firstIter || bestSplit.bestValue > currentLossValue {
			firstIter = false
			bestSplit.bestValue = currentLossValue
			for ind := 0; ind < d; ind++ {
				bestSplit.deltaUp.Set(ind, 0, weightsUp.At(ind, 0))
				bestSplit.deltaDown.Set(ind, 0, weightsDown.At(ind, 0))
			}
			bestSplit.threshold = (em.featuresInter.At(featuresAs[ind], q) + em.featuresInter.At(featuresAs[ind+1], q)) / 2 // TODO: smart threshold selection that assumes correct handling of corner cases
			bestSplit.orderIndex = ind
		}
	}
}

//scanForSplit allocates memory, performs argsort of selected feature column
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

//validateDimansions checks the consistency of dimensions in arrays from the current dataset
//and returns the height (the number of objects), the width (the number of features) and the depth
//(the number of extra features) of the current dataset.
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

//allocateArrays allocates the bias and the raw hessian arrays
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

//Struct for pool of processes
type Pool struct {
	taskChannel chan Runnable
	waitGroup   sync.WaitGroup
}

//Thread of executing tasks
func (pool *Pool) executor() {
	for {
		if task, ok := <-pool.taskChannel; ok {
			task.Run()
			pool.waitGroup.Done()
		} else {
			return
		}
	}
}

//Interface for tasks
type Runnable interface {
	Run()
}

//Create new pool for n async processes
func NewPool(n int) (ret *Pool) {
	ret = &Pool{}
	ret.taskChannel = make(chan Runnable)
	for t := 0; t < n; t++ {
		go ret.executor()
	}
	return
}

//Add task in pool
func (pool *Pool) AddTask(task Runnable) {
	pool.waitGroup.Add(1)
	pool.taskChannel <- task
}

//Wait for all task finish
func (pool *Pool) WaitAll() {
	pool.waitGroup.Wait()
}

//TaskFindBestSplit is the task of finding the best split to be placed into a multitask Pool
type TaskFindBestSplit struct {
	columnSplits []BestSplit
	ind          int
	f            func(int) BestSplit
}

//Run is the implementation of Runnable interface
func (tfbs *TaskFindBestSplit) Run() {
	tfbs.columnSplits[tfbs.ind] = tfbs.f(tfbs.ind)
}

//TreeNode is a node of a tree. Tree is stored in an array. LeftIndex and RightIndex are euqal to -1
//when the current node is a leaf otherwise they contain array indices of children.
//A leaf node contains LeafIndex that is an index of a LeafNodes array
type TreeNode struct {
	TreeNodeId            int
	FeatureNumber         int
	Threshold             float64
	LeftIndex, RightIndex int // -1, -1 if it is a leaf
	LeafIndex             int // -1 if it is a non-leaf tree node
}

//NewTreeNodeFromSplitInfo creates a new tree node and extract a features index and a split threshold
//from a BestSplit object.
func NewTreeNodeFromSplitInfo(splitInfo BestSplit, treeNodeId int) TreeNode {
	return TreeNode{treeNodeId, splitInfo.featureIndex, splitInfo.threshold, -1, -1, -1}
}

//LeafNode stores leaf-related informatin. It is a prediction from this leaf and possibly some other statistics.
type LeafNode struct {
	LeafNodeId int
	Prediction []float64
}

//NewLeafNode creates a new leaf node
func NewLeafNode(leafData *mat.Dense) (leafNode *LeafNode) {
	h, _ := leafData.Dims()

	leafNode = &LeafNode{-1, make([]float64, h)}

	for ind := 0; ind < h; ind++ {
		leafNode.Prediction[ind] = leafData.At(ind, 0)
	}

	return
}

//OneTree describes one tree in a classifier
type OneTree struct {
	d         int // the extra depth
	TreeNodes []TreeNode
	LeafNodes []LeafNode
}

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

//NewTree builds one new tree in a model
func NewTree(ematrix EMatrix, bias *mat.Dense, regLambda float64, maxDepth int) (oneTree OneTree) {
	oneTree.TreeNodes = make([]TreeNode, 0)
	oneTree.LeafNodes = make([]LeafNode, 0)
	_, oneTree.d = ematrix.featuresExtra.Dims()

	(&oneTree).BuildTree(ematrix, bias, nil, regLambda, maxDepth, 0)

	return
}

//Height calculates the height of a matrix m
func Height(m *mat.Dense) int {
	h, _ := m.Dims()
	return h
}

//BuildTree recurrently builds a tree no
func (oneTree *OneTree) BuildTree(
	ematrix EMatrix, bias *mat.Dense,
	leafInfo *LeafNode, parLambda float64, maxDepth int, currentDepth int,
	) (int, bool) {
	if leafInfo == nil || (currentDepth < maxDepth && Height(ematrix.featuresInter) > 5) { // TODO: More flexible approach to stop condition
		treeNodeId := len(oneTree.TreeNodes)
		log.Printf("\tnodeId = %d size = %d", treeNodeId, Height(ematrix.featuresInter))
		bestSplit := TheBestSplit(ematrix, bias, parLambda)
		currentTreeNode := NewTreeNodeFromSplitInfo(bestSplit, treeNodeId)
		oneTree.TreeNodes = append(oneTree.TreeNodes, currentTreeNode)

		leftEmatrix, rightEmatrix, leftBias, rightBias := ematrix.Split(bias, bestSplit)

		if nodeId, isNode := oneTree.BuildTree(leftEmatrix, leftBias, NewLeafNode(bestSplit.deltaUp), parLambda, maxDepth, currentDepth+1); isNode {
			oneTree.TreeNodes[treeNodeId].LeftIndex = nodeId
		} else {
			oneTree.TreeNodes[treeNodeId].LeafIndex = nodeId
		}

		if nodeId, isNode := oneTree.BuildTree(rightEmatrix, rightBias, NewLeafNode(bestSplit.deltaDown), parLambda, maxDepth, currentDepth+1); isNode {
			oneTree.TreeNodes[treeNodeId].RightIndex = nodeId
		} else {
			oneTree.TreeNodes[treeNodeId].LeafIndex = nodeId
		}

		return treeNodeId, true
	}

	leafNodeId := len(oneTree.LeafNodes)
	leafInfo.LeafNodeId = leafNodeId
	oneTree.LeafNodes = append(oneTree.LeafNodes, *leafInfo)
	return leafNodeId, false
}

//TheBestSplit finds the best possible split in the given ematrix.
//This function performs multithreading iteration over columns of the ematrix.
func TheBestSplit(ematrix EMatrix, bias *mat.Dense, parLambda float64) BestSplit {
	h, w, d := ematrix.validatedDimensions()
	rawHessian := ematrix.allocateArrays()

	taskPool := NewPool(4)
	result := make([]BestSplit, w)

	for q := 0; q < w; q++ {
		bestSplitFunc := func(localQ int) BestSplit {
			return scanForSplit(ematrix, h, d, localQ, bias, LogLoss{}, parLambda, rawHessian)
		}
		taskPool.AddTask(&TaskFindBestSplit{result, q, bestSplitFunc})
	}
	taskPool.WaitAll()

	minimalLoss := 0.0
	bestIndex := 0

	firstTime := true

	for ind, currentSplit := range result {
		if firstTime || minimalLoss > currentSplit.bestValue {
			firstTime = false
			minimalLoss = currentSplit.bestValue
			bestIndex = ind
		}
	}

	return result[bestIndex]
}

//EBooster is the model class
type EBooster struct {
	Trees []OneTree
}

//NewEBooster creates a new model
func NewEBooster(ematrix EMatrix, nStages int, regLambda float64, maxDepth int) (ebooster *EBooster) {
	ebooster = &EBooster{make([]OneTree, 0)}
	h, _ := ematrix.featuresInter.Dims()
	bias := mat.NewDense(h, 1, nil)
	for stage := 0; stage < nStages; stage++ {
		log.Printf("Tree number %d\n", stage+1)
		tree := NewTree(ematrix, bias, regLambda, maxDepth)
		ebooster.Trees = append(ebooster.Trees, tree)
		deltaB := tree.PredictValue(ematrix.featuresInter, ematrix.featuresExtra)
		bias.Add(bias, deltaB)
	}
	return
}

//PredictValue infers values of target. It requires two sets of features - both interpolating and extrapolating.
func (ebooster EBooster) PredictValue(features_inter, features_extra *mat.Dense) (prediction *mat.Dense) {
	isFirst := true
	for _, currentTree := range ebooster.Trees {
		if isFirst {
			isFirst = false
			prediction = currentTree.PredictValue(features_inter, features_extra)
		} else {
			deltaPrediction := currentTree.PredictValue(features_inter, features_extra)
			prediction.Add(prediction, deltaPrediction)
		}
	}
	return
}

func main() {
	ematrix := ReadEMatrix(
		"../prepare_dataset/f_inter.npy",
		"../prepare_dataset/f_extra.npy",
		"../prepare_dataset/target.npy",
	)

	clf := NewEBooster(ematrix, 3, 1e-6, 5)

	prediction := clf.PredictValue(ematrix.featuresInter, ematrix.featuresExtra)

	dst, err := os.Create("a.txt")
	handleError(err)

	h, _ := ematrix.featuresInter.Dims()
	for p := 0; p < h; p++ {
		_, err := dst.WriteString(fmt.Sprintf("%g %g\n", ematrix.target.At(p, 0), prediction.At(p, 0)))
		handleError(err)
	}
	handleError(dst.Close())
}
