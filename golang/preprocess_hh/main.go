package main

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
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
		dest.WriteString(fmt.Sprint(title, "\t", id, "\n"))
	}
	fmt.Println(len(title2id), " records are written")
}

func consciseMerge(flnm string, title2id *map[string]int, dest *os.File) (ok bool){
	var err error
	source, err := os.Open(flnm)
	if err != nil {
		return false
	}

	// baseTime := time.Date(2016, 01, 01, 0, 0, 0, 0, time.UTC)
	lineReader := bufio.NewReader(source)
	lineMeter := 0
	for {
		line, _, err := lineReader.ReadLine()
		if err != nil {
			break
		}
		tokens := bytes.Split(line, []byte("\t"))
		// dayTime := tokens[1]
		title := bytes.ToLower(tokens[2])
		target := tokens[3]
		weight := tokens[4]

		// currentTime, err := time.Parse("2016-01-01", string(dayTime))
		// timeDiff := currentTime.Sub(baseTime)
		// deltaDays := int(timeDiff.Hours() / 24 + 1e-5)
		titleId, ok := (*title2id)[string(title)]
		if !ok {
			panic("there is no such title in the dictionary: " + string(title))
		}
		dest.WriteString(fmt.Sprintf("%d %d %s %s\n", lineMeter, titleId, target, weight))
		lineMeter++
	}
	return true
}

func compactTable() {
	var source *os.File
	var err error
	if source, err = os.Open("/mnt/database/golang_process/title2id.csv"); err != nil {
		log.Panic(err)
	}
	lineReader := bufio.NewReader(source)
	title2id := make(map[string]int)
	for {
		line, _, err := lineReader.ReadLine()
		if err != nil {
			break
		}
		parts := bytes.Split(line, []byte("\t"))
		title := parts[0]
		var id int
		fmt.Sscanf(string(parts[1]), "%d", &id)
		title2id[string(title)] = id
	}

	dest, err := os.Create("/mnt/database/golang_process/compact_table.csv")
	if err != nil {
		log.Panic(err)
	}
	fileInd := 0

	for {
		prefix := "/mnt/database/csvs/tsv_presto/"
		suffix := fmt.Sprintf("%04d.csv", fileInd)
		ok := consciseMerge(prefix+suffix, &title2id, dest)
		fmt.Println(prefix + suffix)
		if !ok {
			break
		}
		fileInd++
	}
}

func main() {
	compactTable()
}
