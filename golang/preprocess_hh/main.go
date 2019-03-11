package main

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
	"sort"
)

func processOneFile(flnm string, title2id *map[string]int) (records int) {
	var file *os.File
	var err error
	if file, err = os.Open(flnm); err != nil {
		// log.Fatal(err)
		return 0
	}

	lineReader := bufio.NewReader(file)
	for {
		line, _, err := lineReader.ReadLine()
		if err != nil {
			break
		}
		tokens := bytes.Split(line, []byte("\t"))
		for ind := range tokens {
			tokens[ind] = bytes.ToLower(tokens[ind])
		}
		title := tokens[2]
		if _, ok := (*title2id)[string(title)]; !ok {
			(*title2id)[string(title)] = len(*title2id)
		}
	}
	return len(*title2id)
}

func fillTitles2Id() {
	title2id := make(map[string]int)

	fileInd := 0

	for {
		prefix := "/mnt/database/csvs/tsv_presto/"
		suffix := fmt.Sprintf("%04d.csv", fileInd)
		newTitles := processOneFile(prefix+suffix, &title2id)
		fmt.Println(prefix + suffix)
		if newTitles == 0 {
			break
		}
		fileInd++
	}

	var dest *os.File
	var err error
	if dest, err = os.Create("/mnt/database/golang_process/title2id.csv"); err != nil {
		log.Fatal(err)
	}
	for title, id := range title2id {
		_, err := dest.WriteString(fmt.Sprint(title, "\t", id, "\n"))
		if err != nil {
			log.Panic(err)
		}
	}
	fmt.Println(len(title2id), " records are written")
}

func consciseMerge(flnm string, title2id *map[string]int, dest *os.File, fileInd int) (ok bool){
	var err error
	source, err := os.Open(flnm)
	if err != nil {
		return false
	}

	lineReader := bufio.NewReader(source)
	lineMeter := 0
	for {
		line, _, err := lineReader.ReadLine()
		if err != nil {
			break
		}
		tokens := bytes.Split(line, []byte("\t"))
		title := bytes.ToLower(tokens[2])
		target := tokens[3]
		weight := tokens[4]

		titleId, ok := (*title2id)[string(title)]
		if !ok {
			panic("there is no such title in the dictionary: " + string(title))
		}
		_, err = dest.WriteString(fmt.Sprintf("%d %d %s %s\n", fileInd, titleId, target, weight))
		if err != nil {
			log.Panic(err)
		}
		lineMeter++
	}
	return true
}

func loadTitle2id(flnm string) (title2id map[string]int) {
	var source *os.File
	var err error
	if source, err = os.Open(flnm); err != nil {
		log.Panic(err)
	}
	lineReader := bufio.NewReader(source)
	title2id = make(map[string]int)
	for {
		line, _, err := lineReader.ReadLine()
		if err != nil {
			break
		}
		parts := bytes.Split(line, []byte("\t"))
		title := parts[0]
		var id int
		_, err = fmt.Sscanf(string(parts[1]), "%d", &id)
		if err != nil {
			log.Panic(err)
		}
		title2id[string(title)] = id
	}
	return
}

func compactTable() {
	workDir := "/mnt/database/golang_process/"
	title2id := loadTitle2id(workDir + "title2id.csv")
	dest, err := os.Create(workDir + "compact_table.csv")
	if err != nil {
		log.Panic(err)
	}

	fileInd := 0
	for {
		prefix := "/mnt/database/csvs/tsv_presto/"
		suffix := fmt.Sprintf("%04d.csv", fileInd)
		ok := consciseMerge(prefix+suffix, &title2id, dest, fileInd)
		fmt.Println(prefix + suffix)
		if !ok {
			break
		}
		fileInd++
	}
}

type Weights struct {
	ZeroWeight int
	OneWeight int
}

func twoColumn() {
	workDir := "/mnt/database/golang_process/"
	sourceSuff := "compact_table.csv"
	destSuff := "compact_table_two_column.csv"

	flnmSource := workDir + sourceSuff
	flnmDest := workDir + destSuff

	source, err := os.Open(flnmSource)
	if err != nil {
		log.Panic(err)
	}
	defer source.Close()

	dest, err := os.Create(flnmDest)
	if err != nil {
		log.Panic(err)
	}
	defer dest.Close()

	prevDay, currentDay := 0, 0

	dayid2weights := make(map[int]map[int]*Weights)
	lineReader := bufio.NewReader(source)
	for {
		line, _, err := lineReader.ReadLine()
		if err != nil {
			break
		}
		var (
			day int
			titleId int
			target int
			weight int
		)
		fmt.Sscanf(string(line), "%d %d %d %d", &day, &titleId, &target, &weight)
		prevDay = currentDay
		currentDay = day
		if prevDay != currentDay {
			fmt.Print(currentDay, " ")
		}
		_, hasDay := dayid2weights[day]
		if !hasDay {
			dayid2weights[day] = make(map[int]*Weights)
		}
		_, hasStruct := dayid2weights[day][titleId]
		if !hasStruct {
			dayid2weights[day][titleId] = new(Weights)
		}
		s := dayid2weights[day][titleId]
		if target == 0 {
			s.ZeroWeight = weight
		} else {
			s.OneWeight = weight
		}
	}

	daysNumber := len(dayid2weights)
	allDays := make([]int, daysNumber)
	for day, _ := range dayid2weights {
		allDays = append(allDays, day)
	}

	fmt.Println("allDays has length", len(allDays))
	sort.Ints(allDays)
	fmt.Println("head of days: ")
	for ind := 0; ind < 20 && ind < len(allDays); ind ++ {
		fmt.Print(allDays[ind], " ")
	}
	fmt.Println()
	for ind := len(allDays) - 1; ind > len(allDays) - 20 && ind >= 0; ind -- {
		fmt.Print(allDays[ind], " ")
	}
}

func testTwoIndexes() {
	var a map[int]map[int]int
	a = make(map[int]map[int]int)
	a[1] = make(map[int]int)
	a[1][1] = 2

}

func main() {
	twoColumn()
}
