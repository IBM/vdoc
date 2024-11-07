# VDoc
Code of the emnlp paper "More Bang for your Context: Virtual Documents for Question Answering over Long Documents"
Please cite: [to fill later]

## Installation
1. Start by creating a local Python 3.11 environment using `pyenv` and `venv` or `conda`.  For `pyenv`, follow the instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv), and then run: 

```bash
pyenv install 3.11
pyenv local 3.11
pyenv shell 3.11
```

2. Clone this repository by 
```bash
git clone git@github.com:IBM/vdoc.git
cd vdoc
```

3. Create a `virtual env` or `conda env`. For `virtual env`, use:

```bash
python3.11 -m venv venv
source ./venv/bin/activate
```

4. Install the requirements:
```bash
pip install -r requirements.txt
```

5. Add root directory to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

6. Download the [scrolls](https://www.scrolls-benchmark.com/tasks) datasets.

## Run the vdoc 

An example usage is under create_virtual_document.main()

## Run the EMNLP-paper Direct Vdoc Evaluation (*)

Try the [notebook](vdoc_on_scrolls.ipynb) to run a direct vdoc evaluation on qasper, using
window size of 2048 tokens. 

The script
```
./scripts/run_all_direct_vdoc.sh
```
iterates over various window sizes and various rankers. Change the HOME variable to point to your downloaded files 

Alternatively, you can run directly 
```bash
python -u vdoc --input_file --output_file --model_name --model_token_limit --max_new_tokens --passage_len --order
```

where input_file points to the appropriate validation.jsonl file of Qasper or NarrativeQA

The output_file is a .csv that contains the generated vdoc for each input example, under the `content` field. It can be used to run inference with the vdoc.

The code prints results of the direct vdoc evaluation.

```
#Total count of examples
#Total vdoc activations
#Bad vdocs 
```

(*) The current code does not include the semantic segmentation on GoogleNQ (last column in Table 2). The semantic segmentation is an internal IBM code that is not released as part of this open source.
You can still run the sliding-window mode on GoogleNQ (by passing --dataset googlenq) to vdoc.py.