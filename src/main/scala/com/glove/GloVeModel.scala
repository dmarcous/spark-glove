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
import java.io.File
import java.io.FileOutputStream
import java.io.ObjectOutputStream

object GloVeModel
{
  val VOCAB_FILE = "vocab.hashmap"
  val VECTORS_FILE = "vectors.hashmap" 
}

case class GloVeModel(vocabulary: HashMap[String, Int], wordVectors: HashMap[String, Vector])
{   
  def transform(word: String): Vector = 
  {
    GloVeModelOperations.transform(word, vocabulary, wordVectors)
  }
  
  def getTopKSimilarWords(word: String, K: Int): Seq[String] = 
  {
    GloVeModelOperations.getTopKSimilarWords(word, vocabulary, wordVectors, K)
  }
  
  def getTopKSimilarAnalogies(word1: String, word2: String, targetWord1: String, K: Int): Seq[String] = 
  {
    GloVeModelOperations.getTopKSimilarAnalogies(word1, word2, targetWord1, vocabulary, wordVectors, K)
  }
  
  def save(path: String) =
  {
    val file = new File(path)
    file.mkdirs

    val vocabFile = new File(path, GloVeModel.VOCAB_FILE)
    val f_vocab = new FileOutputStream(vocabFile)
    val s_vocab = new ObjectOutputStream(f_vocab)
    s_vocab.writeObject(vocabulary);
    s_vocab.close();
    
    val vectorsFile = new File(path, GloVeModel.VOCAB_FILE)
    val f_vectors = new FileOutputStream(vectorsFile)
    val s_vectors = new ObjectOutputStream(f_vectors)
    s_vectors.writeObject(wordVectors);
    s_vectors.close();
  }
}