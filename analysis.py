# Import necessary libraries for analysis and visualization
# You might need to run pip install for some of the libaries below
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

'''Remove all code related to df2 if you only run the model detection once.'''

# Create dataframes for MYSSP_detection_result.csv and MYSSP_detection_result_2.csv
# Change "MYSSP_detection_result.csv" and "MYSSP_detection_result_2.csv" to the name of your file(s).
df1 = pd.read_csv("MYSSP_detection_result.csv")
df2 = pd.read_csv("MYSSP_detection_result_2.csv")

# Print out df1
# Note that df1 corresponds to the RCNN model with a confidence threshold of 0.4. For simplicity, we will refer to it as Model 1.
df1

# Print out df2
# df2 corresponds to the RCNN model with a confidence threshold of 0.2. For simplicity, we will refer to it as Model 2.
df2

# A rough idea on how the detector performs (Model 1)
print(pd.crosstab(df1['Object'], df1['Detection output']))

# A rough idea on how the detector performs (Model 2)
print(pd.crosstab(df2['Object'], df2['Detection output']))

# Encode the objects with a numeric label
object_mapping = {
    "Book1": 1,
    "Book2": 1,
    "Book3": 1,
    "remote1": 2,
    "remote2": 2,
    "remote3": 2,
    "spoon1": 3,
    "spoon2": 3,
    "spoon3": 3,
}

df1["Object_encoded"] = df1["Object"].map(object_mapping)
df2["Object_encoded"] = df2["Object"].map(object_mapping)

# Encode the detection output with reference to the objects
detection_mapping = {
    "No detection": 0,
    "book": 1,
    "remote control": 2,
    "spoon": 3,
}

df1["Detection_encoded"] = df1["Detection output"].map(detection_mapping)
df2["Detection_encoded"] = df2["Detection output"].map(detection_mapping)

# Define a function to calculate the overall accuracy
def overall_accuracy(df):
    df = df.copy()
    
    y = df["Object_encoded"] == df["Detection_encoded"]
    acc = y.mean() * 100
    
    print(f"Overall accuracy: {acc:.2f}%")

# Overall accuracy of Model 1
overall_accuracy(df1)

# Overall accuracy of Model 2
overall_accuracy(df2)

# Count the frequency of each detection output in Model 1
df1["Detection output"].value_counts()

# Count the frequency of each detection output in Model 2
df2["Detection output"].value_counts()

# Define a function to generate a confusion matrix
def plot_confusion_matrix(df):
    y_true = df["Object_encoded"]
    y_predicted = df["Detection_encoded"]

    class_labels = [0, 1, 2, 3]
    class_names = ["No detection", "Book", "Remote", "Spoon"]

    cm = confusion_matrix(
        y_true,
        y_predicted,
        labels=class_labels
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Plot out the confusion matrix of Model 1
plot_confusion_matrix(df1)

# Plot out the confusion matrix of Model 2
plot_confusion_matrix(df2)

# Define a function to calculate accuracy of the model for each feature
def accuracy_by_feature(df,feature):
    df = df.copy()

    df["Correct"] = (
        df["Object_encoded"] == df["Detection_encoded"]
    )
    
    accuracy = (df.groupby(feature)["Correct"].mean().mul(100).reset_index(name="Accuracy (%)"))

    return accuracy

# Accuracy by features of Model 1
features = ["Object", "Percentage", "Orientation", "Colour", "Background"]

print("\nThe results for the initial dataframe without lowering the confidence threshold are as follows: \n")

for feature in features:
    print(f"\nAccuracy by {feature}:")
    display(accuracy_by_feature(df1, feature))

# Accuracy by features of Model 2
features = ["Object", "Percentage", "Orientation", "Colour", "Background"]

print("\nThe results for the dataframe after lowering the confidence threshold are as follows: \n")

for feature in features:
    print(f"\nAccuracy by {feature}:")
    display(accuracy_by_feature(df2, feature))