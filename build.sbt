name := "spark-glove"

organization := "com.glove"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.1"

resolvers ++= Seq(
  "All Spark Repository -> bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
)

libraryDependencies ++= Seq(
  "org.apache.spark"      %% "spark-core"       % sparkVersion,
  "org.apache.spark"      %% "spark-sql"        % sparkVersion,
  "org.apache.spark"      %% "spark-mllib"      % sparkVersion,
  "org.apache.spark"      %% "spark-catalyst"   % sparkVersion
)

