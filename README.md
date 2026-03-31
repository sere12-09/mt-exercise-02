# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marcamsler1/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh



# MT Exercise 02 

In this MT exercise, I trained a recurrent neural network language model using the PyTorch example code provided in the repo and analyzed how different dropout values influence the performance of the model.


## Task 1

For this task, I used a custom dataset based on Brothers Grimm fairy tales. The dataset was prepared using the provided script:

./scripts/download_data.sh

The model was then trained with:

python tools/pytorch-examples/word_language_model/main.py \
  --data data/grimm \
  --epochs 5

After training, sample text was generated using:

python tools/pytorch-examples/word_language_model/generate.py \
  --data data/grimm \
  --checkpoint models/model.pt \
  --outf samples/sample_grimm.txt

This produces a file containing generated text based on the model.


## Task 2

The main change introduced in the code was the addition of a new argument `--log_file` in the `main_modified.py` script. This allows the model to save training, validation and test perplexity values. These log files are later used to create tables and plots for analysis.

To train a model, the following command was used, for example for the dropout 0.0:

python tools/pytorch-examples/word_language_model/main.py \
  --data tools/pytorch-examples/word_language_model/data/wikitext-2 \
  --dropout 0.0 \
  --emsize 200 \
  --nhid 200 \
  --epochs 5 \
  --log_file logs/dropout_0.0.tsv

I trained multiple models with different dropout values, including 0.0 (no dropout) and higher values up to 0.8. Each model produces a separate log file stored in the `logs` directory.


## 2 Plots and 3 Tables

I created a `plot.py` script which reads all log files, extracts the perplexity values and generates line plots for both training and validation perplexity. It also constructs tables that are used in the PDF report.

By using the `python plot.py`command I get 5 output files in total:

- 3 tables: `test_table.csv`, `training_table.csv`,`validation_table.csv` (included in the PDF report)
- 2 plots: `training_plot.png`, `validation_plot.png` (included in the PDF report)


## Notes

The directories `logs/`, `models/`, `samples/` etc. are not included in this repository, however the results derived from these files (tables and plots text) are included in the final PDF report.

The modified version of `main.py` with logging functionality is included in the file `main_modified.py`.