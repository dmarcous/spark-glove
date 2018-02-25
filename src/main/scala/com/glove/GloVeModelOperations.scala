package com.glove

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.PairRDDFunctions
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.PairRDDFunctions
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.Dataset
import org.apache.spark.rdd.EmptyRDD
import scala.math
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.DenseVector
import scala.util.Random
import java.io.File
import java.io.FileInputStream
import java.io.ObjectInputStream

object GloVeModelOperations
{
  
  val LEARNING_RATE=0.05 // Initial learning rate for GloVe training
  val NUM_FIT_PARTITIONS=15
  val RAND_MAX = 32767
  
  def fit(@transient spark: SparkSession, words: Dataset[String], VOCAB_MIN_COUNT: Int, VECTOR_SIZE: Int, MAX_ITER: Int, WINDOW_SIZE: Int, X_MAX: Int, ALPHA: Double): (GloVeModel) =
  {
    val (vocabulary, wordVectors) = 
      this.createVectorRepresentations(spark, words, VOCAB_MIN_COUNT, VECTOR_SIZE, MAX_ITER, WINDOW_SIZE, X_MAX, ALPHA)
      
    new GloVeModel(vocabulary, wordVectors)
  }
  
  def createVectorRepresentations(@transient spark: SparkSession, words: Dataset[String], VOCAB_MIN_COUNT: Int, VECTOR_SIZE: Int, MAX_ITER: Int, WINDOW_SIZE: Int, X_MAX: Int, ALPHA: Double): (HashMap[String, Int], HashMap[String, Vector]) = 
  { 
    // Compute vocabulary out of loaded word corpus
    val cleanWords = Vocabulary.cleanWordList(spark, words)
    val initialVocabulary = Vocabulary.computeVocabulary(spark, cleanWords, VOCAB_MIN_COUNT)
    val vocabulary = spark.sparkContext.broadcast(initialVocabulary)
    
    // Compute cooccurrence matrix
    val cooccurrenceMatrixFull = Cooccur.computeCooccurrenceMatrix(spark, cleanWords, vocabulary, WINDOW_SIZE)
    val cooccurrenceMatrix = cooccurrenceMatrixFull.coalesce(NUM_FIT_PARTITIONS, false)
    
    // Initialize vectors and gradients for fitting iterations
    var wordVectorsObject : HashMap[String, Vector] = new HashMap[String, Vector]()
    var wordVectorBiasesObject : HashMap[String, Double] = new HashMap[String, Double]()
    var wordGradientsObject : HashMap[String, Vector] = new HashMap[String, Vector]()
    var wordGradientBiasesObject : HashMap[String, Double] = new HashMap[String, Double]()
    var wordVectors = spark.sparkContext.broadcast(wordVectorsObject)
    var wordVectorBiases = spark.sparkContext.broadcast(wordVectorBiasesObject)
    var wordGradients = spark.sparkContext.broadcast(wordGradientsObject)
    var wordGradientBiases = spark.sparkContext.broadcast(wordGradientBiasesObject)
    
    // train glove
    var iter = 1
    while (iter <= MAX_ITER)
    {
      println("---------------------")
      println("iter : " + iter + "/" + MAX_ITER)
      println("---------------------")
            
      // Fit GloVe model to compute word vectors
      val (wordVectorsCurrent, wordVectorBiasesCurrent, wordGradientsCurrent, wordGradientBiasesCurrent) = 
        this.trainIteration(spark, cooccurrenceMatrix,
                       wordVectors, wordVectorBiases,
                       wordGradients, wordGradientBiases,
                       VECTOR_SIZE, X_MAX, ALPHA)
                             
      wordVectors.unpersist(true)
      wordVectorBiases.unpersist(true)
      wordGradients.unpersist(true)
      wordGradientBiases.unpersist(true)
      wordVectors = spark.sparkContext.broadcast(wordVectorsCurrent)
      wordVectorBiases = spark.sparkContext.broadcast(wordVectorBiasesCurrent)
      wordGradients = spark.sparkContext.broadcast(wordGradientsCurrent)
      wordGradientBiases = spark.sparkContext.broadcast(wordGradientBiasesCurrent)
      
      // Next iteration
      iter = iter + 1
    }
    
    // return word vectors 
    (vocabulary.value, wordVectors.value)
  }
  
  private def trainIteration(@transient spark: SparkSession, cooccurrenceMatrix: RDD[((String, String), Double)],
          wordVectorsCurrent: Broadcast[HashMap[String, Vector]], wordVectorBiasesCurrent: Broadcast[HashMap[String, Double]],
          wordGradientsCurrent: Broadcast[HashMap[String, Vector]], wordGradientBiasesCurrent: Broadcast[HashMap[String, Double]],
          VECTOR_SIZE: Int, X_MAX: Int, ALPHA: Double): 
          (HashMap[String, Vector], HashMap[String, Double],
           HashMap[String, Vector], HashMap[String, Double]) =
  {
    import spark.implicits._
    
    // Go over all partitions on cooccurrence matrix
    val newSavedMaps = 
    cooccurrenceMatrix.mapPartitions{case(coocEntryIter) =>

     // Get broadcast vectors
     var wordVectors = wordVectorsCurrent.value
     var wordVectorBiases = wordVectorBiasesCurrent.value
     var wordGradients = wordGradientsCurrent.value
     var wordGradientBiases = wordGradientBiasesCurrent.value
     
       // Go over all pairs on cooccurrence matrix
       coocEntryIter.foreach{case((word1, word2), count) =>
             
        // Get current word vectors and gradients
        val vec1 = wordVectors.getOrElse(word1, Vectors.dense(Array.fill[Double](VECTOR_SIZE)(((Random.nextInt(RAND_MAX)/RAND_MAX.toDouble)-0.5)/VECTOR_SIZE))) // random init missing vecs
        val vec2 = wordVectors.getOrElse(word2, Vectors.dense(Array.fill[Double](VECTOR_SIZE)(((Random.nextInt(RAND_MAX)/RAND_MAX.toDouble)-0.5)/VECTOR_SIZE)))
        var word1Bias = wordVectorBiases.getOrElse(word1, 0.0) // init word bias to 0.0
        var word2Bias = wordVectorBiases.getOrElse(word2, 0.0)
        var word1Gradients = wordGradients.getOrElse(word1, Vectors.dense(Array.fill[Double](VECTOR_SIZE)(1.0))) // init gradients to 1.0
        var word2Gradients = wordGradients.getOrElse(word2, Vectors.dense(Array.fill[Double](VECTOR_SIZE)(1.0))) 
        var word1BiasGradient = wordGradientBiases.getOrElse(word1, 1.0) // init gradient biases to 1.0
        var word2BiasGradient = wordGradientBiases.getOrElse(word2, 1.0)

        /* Calculate loss */
        val product = this.dotProduct(vec1, vec2)
        val unbaiasedProduct = product + word1Bias + word2Bias
        val weightingValue = math.pow(math.min(1.0, count/X_MAX), ALPHA) // formula 9 in the article - simplified
        val loss = weightingValue * (unbaiasedProduct - math.log(count))

        /* Adaptive gradient updates */
        val learningLoss = loss * LEARNING_RATE // =fdiff
        var word1SumUpdates = 0.0
        var word2SumUpdates = 0.0
        var word1Updates = Array.ofDim[Double](VECTOR_SIZE) //ListBuffer.fill(VECTOR_SIZE)(0.0)
        var word2Updates = Array.ofDim[Double](VECTOR_SIZE) //ListBuffer.fill(VECTOR_SIZE)(0.0)
        var word1GradientsUpdates = word1Gradients.toArray
        var word2GradientsUpdates = word2Gradients.toArray

        // Go over word vectors
        Range(0,VECTOR_SIZE).foreach{case(factor) =>
          // learning rate times gradient for word vectors
          val updatedFactorValue1 = learningLoss * vec2(factor) // =temp1
          val updatedFactorValue2 = learningLoss * vec1(factor) // =temp2
          // adaptive updates
          word1Updates(factor) = updatedFactorValue1 / math.sqrt(word1GradientsUpdates(factor))
          word2Updates(factor) = updatedFactorValue2 / math.sqrt(word2GradientsUpdates(factor))
          word1SumUpdates += word1Updates(factor)
          word2SumUpdates += word2Updates(factor)
          word1GradientsUpdates(factor) += math.pow(updatedFactorValue1, 2.0)
          word2GradientsUpdates(factor) += math.pow(updatedFactorValue2, 2.0)
        }

        wordGradients.put(word1, Vectors.dense(word1GradientsUpdates))
        wordGradients.put(word2, Vectors.dense(word2GradientsUpdates))
        
        // Compute updated word vectors
        if(!word1SumUpdates.isNaN() && !word1SumUpdates.isInfinite() && !word2SumUpdates.isNaN() && !word2SumUpdates.isInfinite())
        {
          val vec1_updates = Vectors.dense(vec1.toDense.values.zip(word1Updates).map{case(factor, update) => (factor - update)})
          val vec2_updates = Vectors.dense(vec2.toDense.values.zip(word2Updates).map{case(factor, update) => (factor - update)})
          wordVectors.put(word1, vec1_updates)
          wordVectors.put(word2, vec2_updates)
        }
        else 
        {
          wordVectors.put(word1, vec1)
          wordVectors.put(word2, vec2)
        }
        
        // Update bias values
        val word1BiasUpdateTerm = learningLoss / math.sqrt(word1BiasGradient)
        if (!word1BiasUpdateTerm.isNaN & !word1BiasUpdateTerm.isInfinite()) word1Bias -= word1BiasUpdateTerm
        val word2BiasUpdateTerm = learningLoss / math.sqrt(word2BiasGradient)
        if (!word2BiasUpdateTerm.isNaN & !word2BiasUpdateTerm.isInfinite()) word2Bias -= word2BiasUpdateTerm
        val squaredLearningLoss = math.pow(learningLoss, 2.0)
        word1BiasGradient += squaredLearningLoss
        word2BiasGradient += squaredLearningLoss
        // Update biases in hashmaps
        wordVectorBiases.put(word1, word1Bias)
        wordVectorBiases.put(word2, word2Bias)
        wordGradientBiases.put(word1, word1BiasGradient)
        wordGradientBiases.put(word2, word2BiasGradient)
        
     }
   
     (wordVectors.iterator.zip(wordVectorBiases.iterator).zip(
      wordGradients.iterator).zip(wordGradientBiases.iterator))
    }
    
    val reducedMaps = newSavedMaps.collect
    
    val updatedWordVectors = HashMap(reducedMaps.map(_._1._1._1): _*)
    val updatedWordVectorBiases = HashMap(newSavedMaps.collect.map(_._1._1._2): _*)
    val updatedWordGradients = HashMap(newSavedMaps.collect.map(_._1._2): _*)
    val updatedWordGradientBiases = HashMap(newSavedMaps.collect.map(_._2): _*)
    
    (updatedWordVectors, updatedWordVectorBiases,
     updatedWordGradients, updatedWordGradientBiases)
  }
  
  def transform(word: String, vocabulary: HashMap[String, Int], wordVectors: HashMap[String, Vector]): Vector = {
    vocabulary.get(word) match {
      case Some(occurences: Int) =>
        wordVectors.get(word) match {
          case Some(wordVector: Vector) =>
            wordVector
          case _ =>
            throw new IllegalStateException("wordVector for word - " + word + " does not exist")
        }
      case _ =>
        throw new IllegalStateException("word - " + word + " not in vocabulary")
    }
  }
  
  def getTopKSimilarWords(word: String, vocabulary: HashMap[String, Int], wordVectors: HashMap[String, Vector], K: Int): Seq[String] = 
  {
    val wordVector = this.transform(word, vocabulary, wordVectors)
    val topKSimilarWords = 
      wordVectors
      .filterKeys(_!=word)
      .mapValues{case(otherVector) => this.cosineDistance(wordVector, otherVector)}
      .toSeq.sortBy(_._2)
      .take(K)
      .map(_._1)
    
    topKSimilarWords
  }
  
  def getTopKSimilarAnalogies(word1: String, word2: String, targetWord1: String, vocabulary: HashMap[String, Int], wordVectors: HashMap[String, Vector], K: Int): Seq[String] = 
  {
    val wordVectorW1 = this.transform(word1, vocabulary, wordVectors)
    val wordVectorW2 = this.transform(word2, vocabulary, wordVectors)
    val wordVectorT1 = this.transform(targetWord1, vocabulary, wordVectors)
    val wordVector =
      Vectors.dense(
      wordVectorW1.toDense.values
      .zip(wordVectorW2.toDense.values)
      .zip(wordVectorT1.toDense.values)
      .map{case((w1, w2), t1) => (w2 - w1 + t1)})
    
    val topKSimilarAnalogyWords = 
      wordVectors
      .filterKeys{!Seq(word1, word2, targetWord1).contains(_)}
      .mapValues{case(otherVector) => this.cosineDistance(wordVector, otherVector)}
      .toSeq.sortBy(_._2)
      .take(K)
      .map(_._1)
    
    topKSimilarAnalogyWords
  }
  
  private def cosineDistance(vec1: Vector, vec2: Vector): Double = 
  {
    val magnitude1 = Vectors.norm(vec1, 2.0)
    val magnitude2 = Vectors.norm(vec1, 2.0)
    val denom = magnitude1 * magnitude2

    if(denom == 0.0)
    {
        0.0
    } 
    else 
    {
        1 - this.dotProduct(vec1, vec2)/denom
    }
  }
  
  private def dotProduct(vec1: Vector, vec2: Vector): Double =
  {
    val product = 
      vec1.toDense.values.zip(vec2.toDense.values)
      .map{case(factor1, factor2) => factor1*factor2}
      .reduce(_+_)
    product
  }
  
  def save(model: GloVeModel, path: String) =
  {
    model.save(path)
  }
  
  def load(path: String): GloVeModel =
  {
    val vocabFile = new File(path, GloVeModel.VOCAB_FILE )
    val f_vocab = new FileInputStream(vocabFile);
    val s_vocab = new ObjectInputStream(f_vocab);
    val vocabulary = s_vocab.readObject.asInstanceOf[HashMap[String, Int]];
    s_vocab.close();
    
    val vectorsFile = new File(path, GloVeModel.VECTORS_FILE)
    val f_vectors = new FileInputStream(vectorsFile)
    val s_vectors = new ObjectInputStream(f_vectors)
    val wordVectors = s_vectors.readObject.asInstanceOf[HashMap[String, Vector]];
    s_vectors.close();
    
    new GloVeModel(vocabulary, wordVectors)
  }
  
}