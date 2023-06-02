import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Creating PySpark session
spark = SparkSession.builder.appName("Indian Liver Patient Detection").getOrCreate()

# Load the saved model
model_path = "./chem_class"
model = RandomForestClassificationModel.load(model_path)

# Define the chemical classifier function
def chem_classifier(data):
    # Perform the chemical classifier using the loaded model
    prediction = model.transform(data).select("prediction").collect()[0][0]
    return prediction


def main():
    # Set the page title
    st.title('Liver Patient Classifier')

    # Create input fields for liver patient features
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    if Gender == 'Male':
        Gender = 0
    else:
        Gender = 1
    Age = st.slider('Age', 1, 100)
    Total_Bilirubin = st.slider('Total Bilirubin', 0.4, 75.0)
    Direct_Bilirubin = st.slider('Direct Bilirubin', 0.1, 20.0)
    Alkaline_Phosphotase = st.slider('Alkaline Phosphotase', 63, 2110)
    Alamine_Aminotransferase = st.slider('Alamine Aminotransferase', 10, 2000)
    Aspartate_Aminotransferase = st.slider('Aspartate Aminotransferase', 10, 5000)
    Total_Protiens = st.slider('Total Protiens', 2.5, 10.0)
    Albumin = st.slider('Albumin', 0.5, 6.0)
    Albumin_and_Globulin_Ratio = st.slider('Albumin and Globulin Ratio', 0.1, 3.0)

    # Create a button to perform the prediction
    if st.button('Classify'):

        # Define the schema for the DataFrame
        schema = StructType([
        StructField("Age", IntegerType(), nullable=True),
        StructField("Gender", IntegerType(), nullable=False),
        StructField("Total_Bilirubin", DoubleType(), nullable=True),
        StructField("Direct_Bilirubin", DoubleType(), nullable=True),
        StructField("Alkaline_Phosphotase", IntegerType(), nullable=True),
        StructField("Alamine_Aminotransferase", IntegerType(), nullable=True),
        StructField("Aspartate_Aminotransferase", IntegerType(), nullable=True),
        StructField("Total_Protiens", DoubleType(), nullable=True),
        StructField("Albumin", DoubleType(), nullable=True),
        StructField("Albumin_and_Globulin_Ratio", DoubleType(), nullable=True)

        ])
        
        # Create a dictionary with the input values
        data = {
            'Age': [Age],
            'Gender': [Gender],
            'Total_Bilirubin': [Total_Bilirubin],
            'Direct_Bilirubin': [Direct_Bilirubin],
            'Alkaline_Phosphotase': [Alkaline_Phosphotase],
            'Alamine_Aminotransferase': [Alamine_Aminotransferase],
            'Aspartate_Aminotransferase': [Aspartate_Aminotransferase],
            'Total_Protiens': [Total_Protiens],
            'Albumin': [Albumin],
            'Albumin_and_Globulin_Ratio': [Albumin_and_Globulin_Ratio]
        }

        # Convert the dictionary to a list of tuples
        rows = list(zip(data["Age"], data["Gender"], data["Total_Bilirubin"], data["Direct_Bilirubin"], data["Alkaline_Phosphotase"], data["Alamine_Aminotransferase"], data["Aspartate_Aminotransferase"], data["Total_Protiens"], data["Albumin"], data["Albumin_and_Globulin_Ratio"]))

        # Create the DataFrame
        df = spark.createDataFrame(rows, schema)

        # Perform feature vectorization using VectorAssembler
        vector_assembler = VectorAssembler(inputCols=["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"],
                                           outputCol="features")

        input_df = vector_assembler.transform(df)

        # Perform the chemical classifier using the loaded model
        prediction = chem_classifier(input_df)
        if prediction == 1:
            st.warning('Sorry, Patient has liver disease.')
        else:
            st.success('Congrats!, Patient has no liver disease.')


if __name__ == '__main__':
    main()