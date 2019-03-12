package main

import "fmt"

type Point struct {
	X int
	Y int
}

func main() {
	var a map[int]*Point
	a = make(map[int]*Point)
	a[1] = &Point{10, 20}
	a[1].X = 100
	fmt.Print(a[1])
}
