package com.glove

import org.apache.spark.sql.SparkSession

object GloVeRunnerTest
{
  def main(args: Array[String])
  {
    // Create spark context
    val appName="GloVeRunner"
    val spark = 
     SparkSession
     .builder
     .master("local")
     .appName(appName)
     .getOrCreate()
    import spark.implicits._
    
    // Default params
    val DEFAULT_VOCAB_MIN_COUNT = 5
    val DEFAULT_VECTOR_SIZE = 50
    val DEFAULT_MAX_ITER = 15
    val DEFAULT_WINDOW_SIZE = 15
    val DEFAULT_X_MAX = 100
    val DEFAULT_ALPHA = 0.75
    
    val inputPath = "src/functional-test/resources/text_spoken_kde/w_spok_2012.txt"
    val VOCAB_MIN_COUNT = DEFAULT_VOCAB_MIN_COUNT
    val VECTOR_SIZE = DEFAULT_VECTOR_SIZE
    val MAX_ITER = DEFAULT_MAX_ITER
    val WINDOW_SIZE = DEFAULT_WINDOW_SIZE
    val X_MAX = DEFAULT_X_MAX
    val ALPHA = DEFAULT_ALPHA
  
    println("inputPath : " + inputPath)
    println("VOCAB_MIN_COUNT : " + VOCAB_MIN_COUNT)
    println("VECTOR_SIZE : " + VECTOR_SIZE)
    println("MAX_ITER : " + MAX_ITER)
    println("WINDOW_SIZE : " + WINDOW_SIZE)
    println("X_MAX : " + X_MAX)
    println("ALPHA : " + ALPHA)
    
    // Read input file
    val words = 
      spark.read
      .textFile(inputPath)
      // split lines to words 
      .flatMap{case(line) => line.split(" ").toSeq}  
          
    // Run GloVe 
    println("Starting GloVe training...")
    val model = 
      GloVeModelOperations.fit(spark, words, VOCAB_MIN_COUNT, VECTOR_SIZE, MAX_ITER, WINDOW_SIZE, X_MAX, ALPHA)
    
    // Print output
    println("Vocabulary : ")
    model.vocabulary.take(5).foreach(println)
    println("Vectors : ")
    model.wordVectors.take(5).foreach(println)
    val word1 = "obama"
    val word2 = "president"
    val wordT = "romney"
    println("From vocab - " + word1 + " : " + model.vocabulary.get(word1).get)
    println("From vocab - " + word2 + " : " + model.vocabulary.get(word2).get)
    println("From vocab - " + wordT + " : " + model.vocabulary.get(wordT).get)
    println("Vector of -obama- : ")
    println(model.transform(word1))
    println("Similar to -obama- : ")
    model.getTopKSimilarWords(word1, 5).foreach(println)
    println("Vector of -president- : ")
    println(model.transform(word2))
    println("Similar to -president- : ")
    model.getTopKSimilarWords(word2, 5).foreach(println)
    println("Vector of -romney- : ")
    println(model.transform(wordT))
    println("Similar to -romney- : ")
    model.getTopKSimilarWords(wordT, 5).foreach(println)
    println("Analogy Obama->President, Romeny->? : ")
    model.getTopKSimilarAnalogies(word1, word2, wordT, 10).foreach(println)

  }
}