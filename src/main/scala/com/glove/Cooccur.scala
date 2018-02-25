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

object Cooccur 
{
  
  val OVERFLOW_LENGTH=1000 // size of cooccurrence buffer to look for context words in
  
  def computeCooccurrenceMatrix(@transient spark: SparkSession, cleanWords: RDD[String], globalVocabulary: Broadcast[HashMap[String, Int]], WINDOW_SIZE: Int): RDD[((String, String), Double)] =
  {
    import spark.implicits._
      
    // Filter only words in vocabulary
    val wordsInVocabulary = 
      cleanWords.filter(globalVocabulary.value.contains(_))
    
    val wordListSize = wordsInVocabulary.count
    val numberOfPartitions = Math.floor(wordListSize/OVERFLOW_LENGTH.toDouble).toInt
    
    val cooccurrenceMatrix =
      wordsInVocabulary.repartition(numberOfPartitions)
      .mapPartitions{case(wordList) =>
        val wordSeq = wordList.toSeq
        val wordsInPartition = wordSeq.size
        
        val partitionMatrixEntries=
        // Go over all words 
        wordSeq.zipWithIndex.flatMap{case(word, wordIndex) =>

          val vocabulary = globalVocabulary.value

          // Compute window sides from both sides
          val loweBound = Math.max(0, wordIndex-WINDOW_SIZE)
          val upperBound = Math.min(wordsInPartition-1, wordIndex+WINDOW_SIZE+1)
          val contextWindowSize = upperBound - loweBound
          if (contextWindowSize > 0 && vocabulary.contains(word))
          {
            // Go over all context words except current word
            Range(loweBound,upperBound+1).map{case (contextIndex) =>
              val contextWord = wordSeq(contextIndex)
              if (vocabulary.contains(contextWord) && contextWord != word)
              {
                val matrixValue = {
                  if (contextIndex < wordIndex) // context before word
                  {
                    ((contextWord, word), 1.0/(wordIndex.toDouble-contextIndex.toDouble))
                  }
                  else // symmetric (context after word)
                  {
                    ((word, contextWord), 1.0/(contextIndex.toDouble-wordIndex.toDouble))
                  }  
                }
                matrixValue
              } else null
            }  
          } else null
        }
        
        // filter null records from break passes (e.g. word not in vocabulary) 
        partitionMatrixEntries.filter{case(matrixRecord) => matrixRecord != null}.toIterator
      }
    
    cooccurrenceMatrix
  }
  
}