package extra_boost_lib

//IntIterable is the interface for iteration over an collection of integers.
type IntIterable interface {
	HasNext() bool
	GetNext() int
}

//Range is an iterator over half interval [begin, end) with the step step.
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
