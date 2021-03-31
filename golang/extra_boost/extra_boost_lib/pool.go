package extra_boost_lib

import "sync"

//Struct for pool of threads
type Pool struct {
	taskChannel chan Runnable
	waitGroup   sync.WaitGroup
}

//A thread of executing tasks.
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

//The interface for tasks.
type Runnable interface {
	Run()
}

//Create a new pool for n async processes.
func NewPool(n int) (ret *Pool) {
	ret = &Pool{}
	ret.taskChannel = make(chan Runnable)
	for t := 0; t < n; t++ {
		go ret.executor()
	}
	return
}

//Add a task to the pool.
func (pool *Pool) AddTask(task Runnable) {
	pool.waitGroup.Add(1)
	pool.taskChannel <- task
}

//Wait for all task to finish.
func (pool *Pool) WaitAll() {
	pool.waitGroup.Wait()
}

//TaskFindBestSplit is the task of finding the best split to be placed into a multitask Pool.
type TaskFindBestSplit struct {
	columnSplits []BestSplit
	ind          int
	f            func(int) BestSplit
}

//Run is an implementation of the Runnable interface.
func (tfbs *TaskFindBestSplit) Run() {
	tfbs.columnSplits[tfbs.ind] = tfbs.f(tfbs.ind)
}
