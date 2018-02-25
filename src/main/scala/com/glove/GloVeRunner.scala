package com.glove

import org.apache.spark.sql.SparkSession

object GloVeRunner
{
  def main(args: Array[String])
  {
    // Create spark context
    val appName="GloVeRunner"
    val spark = 
      SparkSession
     .builder()
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
    val usage = """
    Usage: /usr/lib/spark/bin/spark-submit --class com.glove.GloVeRunner [filename.jar] [inputPath] [VOCAB_MIN_COUNT] [VECTOR_SIZE] [MAX_ITER] [WINDOW_SIZE] [X_MAX] [ALPHA]
    """
    // Input params
    if (args.length == 0)
    {
      println(usage)
      System.exit(1)
    }
    
    val inputPath = args(0)
    val VOCAB_MIN_COUNT = if (args.length > 1) args(1).toInt else DEFAULT_VOCAB_MIN_COUNT
    val VECTOR_SIZE = if (args.length > 2) args(2).toInt else DEFAULT_VECTOR_SIZE
    val MAX_ITER = if (args.length > 3) args(3).toInt else DEFAULT_MAX_ITER
    val WINDOW_SIZE = if (args.length > 4) args(4).toInt else DEFAULT_WINDOW_SIZE
    val X_MAX = if (args.length > 5) args(5).toInt else DEFAULT_X_MAX
    val ALPHA = if (args.length > 6) args(6).toDouble else DEFAULT_ALPHA
  
    if (args.length > 7) 
    {
      println(usage)
      System.exit(1)
    }

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
  }
}