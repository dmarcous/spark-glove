package com.glove

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.PairRDDFunctions
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.PairRDDFunctions
import scala.collection.mutable.HashMap
import scala.io.Source.fromFile

object Vocabulary
{
  val stopWordListPath = "src/main/resources/stop_words.txt"
  
  def computeVocabulary(@transient spark: SparkSession, cleanWords: RDD[String], VOCAB_MIN_COUNT: Int): HashMap[String, Int] =
  {
    import spark.implicits._
      
    val wordCounts = 
      cleanWords
      .map{case(word) => (word, 1)}
      .reduceByKey(_ + _)
      .filter{case(word, appearences) => appearences >= VOCAB_MIN_COUNT} // remove not enough appearances
      .collect()
      
    val vocabulary = HashMap(wordCounts: _*)
    
    vocabulary
  }
  
  def cleanWordList(@transient spark: SparkSession, words: Dataset[String]): RDD[String] = 
  {
    
    import spark.implicits._
    
    val stopWords = hashMapFroWordList(stopWordListPath)
    
    val cleanWords = 
      words.rdd
      .map{case(word) => (word.trim.toLowerCase)}
      .filter{case(word) => word.length() > 1} // remove single chars
      .filter{case(word) => !stopWords.contains(word)} // remove stop words
      .filter{case(word) => word.matches("^[a-zA-Z]*$")} // remove non all alpha chars
      
    cleanWords
  }
  
  private def hashMapFroWordList(filePath: String): HashMap[String, Int] =
  {
    val wordSeq = 
      fromFile(filePath)
      .getLines
      .map { case(word) => (word, 1)} 
      .toSeq
     
    HashMap(wordSeq: _*)
  }
  
}