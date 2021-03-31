package extra_boost_lib

import (
	"gonum.org/v1/gonum/mat"
)

//TreeNode is a node of a tree. Tree is stored in an array. LeftIndex and RightIndex are equal to -1
//when the current node is a leaf otherwise they contain array indices of children.
//A leaf node contains LeafIndex that is an index of the LeafNodes array.
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

//LeafNode stores leaf-related information. It is a prediction from this leaf and possibly some other statistics.
type LeafNode struct {
	LeafNodeId int
	Prediction []float64
}

//NewLeafNode creates a new leaf node.
func NewLeafNode(leafData *mat.Dense, learningRate float64) (leafNode *LeafNode) {
	h, _ := leafData.Dims()

	leafNode = &LeafNode{-1, make([]float64, h)}

	for ind := 0; ind < h; ind++ {
		leafNode.Prediction[ind] = leafData.At(ind, 0) * learningRate
	}

	return
}

//OneTree describes one tree in a classifier.
type OneTree struct {
	d         int // the extra depth
	TreeNodes []TreeNode
	LeafNodes []LeafNode
}

//NewTree builds one new tree in a model.
func NewTree(ematrix EMatrix, bias *mat.Dense, regLambda float64, maxDepth int, learningRate float64, lossKind SplitLoss) (oneTree OneTree) {
	oneTree.TreeNodes = make([]TreeNode, 0)
	oneTree.LeafNodes = make([]LeafNode, 0)
	_, oneTree.d = ematrix.featuresExtra.Dims()

	(&oneTree).BuildTree(ematrix, bias, nil, regLambda, maxDepth, 0, learningRate, lossKind)

	return
}

//BuildTree recurrently builds a tree node.
func (oneTree *OneTree) BuildTree(
	ematrix EMatrix, bias *mat.Dense,
	leafInfo *LeafNode, parLambda float64, maxDepth int, currentDepth int,
	learningRate float64,
	lossKind SplitLoss,
) (int, bool) {
	if leafInfo == nil || (currentDepth < maxDepth && Height(ematrix.featuresInter) > 5) { // TODO: More flexible approach to stop condition
		treeNodeId := len(oneTree.TreeNodes)
		// log.Printf("\tnodeId = %d size = %d", treeNodeId, Height(ematrix.featuresInter))
		bestSplit := TheBestSplit(ematrix, bias, parLambda, lossKind)
		currentTreeNode := NewTreeNodeFromSplitInfo(bestSplit, treeNodeId)
		oneTree.TreeNodes = append(oneTree.TreeNodes, currentTreeNode)

		leftEmatrix, rightEmatrix, leftBias, rightBias := ematrix.Split(bias, bestSplit)

		if nodeId, isNode := oneTree.BuildTree(leftEmatrix, leftBias, NewLeafNode(bestSplit.deltaUp, learningRate), parLambda, maxDepth, currentDepth+1, learningRate, lossKind); isNode {
			oneTree.TreeNodes[treeNodeId].LeftIndex = nodeId
		} else {
			oneTree.TreeNodes[treeNodeId].LeafIndex = nodeId
		}

		if nodeId, isNode := oneTree.BuildTree(rightEmatrix, rightBias, NewLeafNode(bestSplit.deltaDown, learningRate), parLambda, maxDepth, currentDepth+1, learningRate, lossKind); isNode {
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
func TheBestSplit(ematrix EMatrix, bias *mat.Dense, parLambda float64, lossKind SplitLoss) BestSplit {
	h, w, d := ematrix.validatedDimensions()
	rawHessian := ematrix.allocateArrays()

	//log.Printf("ematrix %d\n", h)

	taskPool := NewPool(10)
	result := make([]BestSplit, w)

	for q := 0; q < w; q++ {
		bestSplitFunc := func(localQ int) BestSplit {
			return scanForSplit(ematrix, h, d, localQ, bias, lossKind, parLambda, rawHessian)
		}
		taskPool.AddTask(&TaskFindBestSplit{result, q, bestSplitFunc})
	}
	taskPool.WaitAll()

	minimalLoss := 0.0
	bestIndex := 0

	firstTime := true

	for ind, currentSplit := range result {
		if currentSplit.validSplit && (firstTime || minimalLoss > currentSplit.bestValue) {
			firstTime = false
			minimalLoss = currentSplit.bestValue
			bestIndex = ind
		}
	}

	if firstTime {
		panic("There is no valid split")
	}

	return result[bestIndex]
}
