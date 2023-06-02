from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Imputer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():

    # Creating Pysaprk session:
    spark = SparkSession.builder.appName("Indian Liver Patient Detection").getOrCreate()

    # Provide the path to your CSV file in Google Drive
    csv_path = './Q12.csv'

    # Load Indian Liver Patient Records:
    data_df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Encoding categorical column to numerical format
    data_df = data_df.withColumn("Gender", when(data_df["Gender"] == "Male", 0).otherwise(1))


    # split fatures-target:
    feature_columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
    target_column = 'Dataset'

    # Handle missing values:
    # Create an instance of the Imputer
    imputer = Imputer(inputCols=feature_columns, outputCols=feature_columns)

    # Fit the imputer on the data
    imputer_model = imputer.fit(data_df)

    # Transform the data by replacing null values with mean values
    data_df = imputer_model.transform(data_df)

    # Assemble the features into a vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data_with_features = assembler.transform(data_df).select("features", "Dataset")

    # Split the data into training and testing sets
    (training_data, testing_data) = data_with_features.randomSplit([0.7, 0.3], seed=123)

    # Create an instance of the Random Forest classifier
    rf = RandomForestClassifier(labelCol=target_column, featuresCol="features")

    # Train the model
    model = rf.fit(training_data)

    # Save the trained model
    model_path = "./chem_class"  # Replace with the desired path to save the model
    model.save(model_path)
    print("Model saved successfully!")

    # Make predictions on the testing data
    predictions = model.transform(testing_data)

    # Select the predicted label and true label columns for evaluation
    predicted_labels_df = predictions.select("prediction").collect()[0][0]
    print('Predicted:', predicted_labels_df)
    true_labels_df = predictions.select("Dataset").collect()[0][0]
    print('True:', true_labels_df)

    # Create an instance of MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="Dataset", metricName="accuracy")

    # Calculate the accuracy
    accuracy = evaluator.evaluate(predictions)

    print("Accuracy of the Model:", accuracy*100, '%')


if __name__ == '__main__':
    main()