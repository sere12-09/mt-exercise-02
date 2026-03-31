import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# find all log files inside the logs folder
files = glob.glob("logs/*.tsv")

# read each log file and store it in a dictionary. The key will be the file name without ".tsv"
data = {}

for f in files:
    name = os.path.basename(f).replace(".tsv", "")
    df = pd.read_csv(f, sep="\t")
    data[name] = df

# build the training perplexity table
# For each dropout file, keep only the rows with split == "train"
# Then keep only the columns epoch and perplexity
# Rename the perplexity column to the file name, for example dropout_0.2
train_rows = []

for name, df in data.items():
    train_df = df[df["split"] == "train"][["epoch", "perplexity"]].copy()
    train_df = train_df.rename(columns={"perplexity": name})
    train_rows.append(train_df)

# merge all training tables together by epoch
train_table = train_rows[0]

for df in train_rows[1:]:
    train_table = pd.merge(train_table, df, on="epoch", how="outer")

# sort the training table by epoch and save it
train_table = train_table.sort_values("epoch")
train_table.to_csv("training_table.csv", index=False)

# build the validation perplexity table
# This is the same idea as above, but now we keep split == "valid"
valid_rows = []

for name, df in data.items():
    valid_df = df[df["split"] == "valid"][["epoch", "perplexity"]].copy()
    valid_df = valid_df.rename(columns={"perplexity": name})
    valid_rows.append(valid_df)

# merge all validation tables together by epoch
valid_table = valid_rows[0]

for df in valid_rows[1:]:
    valid_table = pd.merge(valid_table, df, on="epoch", how="outer")

# sort the validation table by epoch and save
valid_table = valid_table.sort_values("epoch")
valid_table.to_csv("validation_table.csv", index=False)

# build the test perplexity table
test_rows = []

for name, df in data.items():
    test_df = df[df["split"] == "test"][["perplexity"]].copy()
    test_value = test_df.iloc[0]["perplexity"]
    test_rows.append({"dropout": name, "test_perplexity": test_value})

# convert the collected test values into a DataFrame, sort it and save it
test_table = pd.DataFrame(test_rows)
test_table = test_table.sort_values("dropout")
test_table.to_csv("test_table.csv", index=False)

# print the three tables in the terminal
print("\nTRAINING TABLE")
print(train_table)

print("\nVALIDATION TABLE")
print(valid_table)

print("\nTEST TABLE")
print(test_table)

# create the validation perplexity plot
# Each line corresponds to one dropout setting
plt.figure()

for name, df in data.items():
    valid_df = df[df["split"] == "valid"]
    plt.plot(valid_df["epoch"], valid_df["perplexity"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Validation Perplexity for different dropout values")
plt.legend()
plt.savefig("validation_plot.png")
plt.close()

# create the training perplexity plot
# Again, each line corresponds to one dropout setting
plt.figure()

for name, df in data.items():
    train_df = df[df["split"] == "train"]
    plt.plot(train_df["epoch"], train_df["perplexity"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Training Perplexity for different dropout values")
plt.legend()
plt.savefig("training_plot.png")
plt.close()

# print the names of the files
print("\nFiles created:")
print("- training_table.csv")
print("- validation_table.csv")
print("- test_table.csv")
print("- training_plot.png")
print("- validation_plot.png")