package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
)

var vdb []VectorDocument

type VectorDocument struct {
	Embedding []float32
	Content   string
}

func main() {
	// start the Ollama server
	go startOllamaServer()

	// add the given document into vdb.gob
	if os.Args[1] == "add" {
		log.Println("adding document:", os.Args[2])
		content, _ := convert(os.Args[2])
		addVectorDocuments(clean(content))
	}

	// loads vector documents from vdb.gob, gets text chunks
	// related to the question, calls the LLM using the chunks
	if os.Args[1] == "call" {
		log.Println("calling model with document")
		loadVdb()
		chunks := getSimilarChunks(os.Args[2])
		call("llama2", strings.Join(chunks, "\n"), os.Args[2])
	}
}

// adds vector documents into the vdb.gob file
func addVectorDocuments(content []string) {
	file, err := os.OpenFile("vdb.gob", os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		log.Println("cannot open gob file:", err)
	}
	defer file.Close()

	embeddings, err := getEmbeddings(content)
	if err != nil {
		log.Println("cannot get embeddings", err)
	}

	for i, c := range content {
		doc := VectorDocument{
			Embedding: embeddings[i],
			Content:   c,
		}
		vdb = append(vdb, doc)
	}
	encoder := gob.NewEncoder(file)
	err = encoder.Encode(vdb)
	if err != nil {
		log.Println("cannot save vdb to file", err)
	}
}

// loads the vdb variable from vdb.gob
func loadVdb() {
	file, err := os.Open("vdb.gob")
	if err != nil {
		log.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&vdb)
	if err != nil {
		log.Println("Error decoding:", err)
		return
	}
	log.Printf("loaded %d records into vdb\n", len(vdb))
}

// converts pdf into text using xpdfreader's pdftotext
func convert(inputpdf string) (string, error) {
	tempdir, err := os.MkdirTemp("", "vdb")
	if err != nil {
		log.Println("unable to create a temporary directory:", err)
		return "", err
	}
	defer os.RemoveAll(tempdir)

	cmd := exec.Command(filepath.Join("bin", "pdftotext"), inputpdf, filepath.Join(tempdir, "output.txt"))
	_, err = cmd.CombinedOutput()
	if err != nil {
		log.Printf("Command error: %s\n", err)
		return "", err
	}

	text, err := os.ReadFile(filepath.Join(tempdir, "output.txt"))
	if err != nil {
		log.Printf("cannot read text: %s\n", err)
		return "", err
	}
	content := string(text)
	content = strings.ToValidUTF8(content, "")
	return content, nil
}

// splits up the content and cleans it up
// by removing duplicates and very chunks
func clean(content string) []string {
	split := strings.Split(content, "\n\n")
	cleaned := []string{}
	for _, s := range split {
		cleaned = append(cleaned, strings.TrimSpace(s))
	}
	unique := removeDuplicates(cleaned)
	shortRemoved := removeShortStrings(unique)
	return shortRemoved
}

func removeDuplicates(s []string) []string {
	m := make(map[string]bool)
	result := []string{}
	for _, item := range s {
		if _, ok := m[item]; !ok {
			m[item] = true
			result = append(result, item)
		}
	}
	return result
}

func removeShortStrings(slice []string) []string {
	var result []string
	for _, str := range slice {
		sl := strings.Split(str, " ")
		if len(sl) > 3 {
			result = append(result, str)
		}
	}
	return result
}

// dot product of 2 float32 slices
func dotproduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}
	var dp float32
	for i := 0; i < len(a); i++ {
		dp += a[i] * b[i]
	}
	return dp
}

// magnitude of a float32 slice
func magnitude(a []float32) float32 {
	var mag float64
	for i := 0; i < len(a); i++ {
		mag += math.Pow(float64(a[i]), 2.0)
	}
	return float32(mag)
}

// cosine similarity of 2 float32 slices
func similarity(a, b []float32) float32 {
	return dotproduct(a, b) / (magnitude(a) * magnitude(b))
}

// get embeddings from Ollama
func getEmbeddings(content []string) ([][]float32, error) {
	llm, err := ollama.New(ollama.WithModel("nomic-embed-text"))
	if err != nil {
		return [][]float32{}, err
	}
	c := context.Background()
	return llm.CreateEmbedding(c, content)
}

// get chunks that are similar to the given question
func getSimilarChunks(question string) []string {
	chunks := make(map[float32]string)
	embedding, _ := getEmbeddings([]string{question})
	for _, doc := range vdb {
		sim := similarity(embedding[0], doc.Embedding)
		chunks[sim] = doc.Content
	}

	// return top 3 chunks
	keys := make([]float32, 0, len(chunks))
	for k := range chunks {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i] > keys[j]
	})
	var topChunks []string
	for _, key := range keys[:3] {
		topChunks = append(topChunks, chunks[key])
	}
	return topChunks
}

// call the Ollama model with the doc and the question
func call(model string, doc string, question string) {
	llm, err := ollama.New(ollama.WithModel(model))
	if err != nil {
		log.Println("Cannot create LLM:", err)
	}
	c := context.Background()
	_, err = llm.GenerateContent(
		c, []llms.MessageContent{
			llms.TextParts(schema.ChatMessageTypeSystem, doc),
			llms.TextParts(schema.ChatMessageTypeHuman, question),
		},
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}), llms.WithMinLength(1024),
	)
	fmt.Println()
	if err != nil {
		log.Println("Cannot generate content:", err)
	}
}
