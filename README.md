# spark-glove

An implementation of GloVe model for learning word representations for big text corpuses distributed with Apache Spark.

Based on the original implementation : https://github.com/stanfordnlp/GloVe

## Details

This project contains a GloVe model representation, train and usage operations, and running examples.

### Steps to train

  1. Reading a corpus into a word Dataset.
  2. Cleaning the datset from sepcial characters, stop words, trimming and lower casing.
  3. Building a vocabulary and filtering low occurence words.
  4. Building a word cooccurrence matrix based on moving windows.
  5. Initializing vectors and gradients data structures.
  6. Run fit iterations to update vector and gradient representations
  7. Creating a GloVeModel object from the vocabulary and trained word vectors.

### Parameters

  1. inputFile - Corpus (Example in resources : w_spok_2012.txt
  2. VOCAB_MIN_COUNT - Minimum word occurences to keep in vocab (default : 5)
  3. VECTOR_SIZE - Word embedding vecotr size (default : 50)
  4. MAX_ITER - Number of fitting iterations (default : 15)
  5. WINDOW_SIZE - Moving window size for building cooccurrence matrix (default : 15)
  6. X_MAX - Training parameter (see article) (default : 100)
  7. ALPHA - Training parameter (see article) (default : 0.75)

### Constants

  1. INITIAL_LEARNING_RATE = 0.05
  2. NUM_FIT_PARTITIONS = 15
  3. OVERFLOW_LENGTH = 1000

## Requirements

Spark 2.1+.

Scala 2.11.

## Usage

### Scala API

```scala
// import 
import com.glove._

// Run GloVe 
println("Starting GloVe training...")
val model = 
	GloVeModelOperations.fit(spark, words, VOCAB_MIN_COUNT, VECTOR_SIZE, MAX_ITER, WINDOW_SIZE, X_MAX, ALPHA)	        

// Save model
GloVeModelOperations.save(model, "/tmp/glove_model")
	    
// Print output
println("Vocabulary : ")
model.vocabulary.take(100).foreach(println)
println("Vectors : ")
model.wordVectors.take(100).foreach(println)

val word1 = "man"
val word2 = "king"
val wordT = "woman"
println("Vector of man : ")
println(model.transform(word1))
println("Similar to man : ")
model.getTopKSimilarWords(word1, 10).foreach(println)
println("Analogy king->man, woman->? : ")
model.getTopKSimilarAnalogies(word1, word2, wordT, 5).foreach(println)

```

### Running GloVe from command line

You can run spark-glove directly form command line using spark-submit.

Parameters stated above

```bash
/usr/lib/spark/bin/spark-submit --class com.glove.GloVeRunner /tmp/glove.jar /tmp/input/w_spok_2012.txt 5 50 15 15 100 0.75
```

## Credits

Written and maintained by :

Daniel Marcous <dmarcous@gmail.com>


