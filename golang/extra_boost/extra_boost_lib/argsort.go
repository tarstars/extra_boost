package extra_boost_lib

import "sort"

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

//Swap swaps elements of the indirect addressing array.
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
