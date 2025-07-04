## Function Induction and Task Generalization

Code for paper "Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition".

### Getting Started

```bash
conda create -n fi4 python=3.12
conda activate fi4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformer_lens==2.16.0 transformers==4.51.3
pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
pip install matplotlib plotly
pip install --upgrade nbformat

# to access data_utils.py and patching_utils.py in subdirectories
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### Key Findings

#### 1. Language models can learn off-by-one addition in context.
* We evaluated six recent language models and they all have non-trivial to near-perfect performance on the task of off-by-one addition.
* See `1_off_by_one_addition_eval/eval.py` for our code to evaluate models on off-by-one addition. `results_aggregated.csv` contains all evaluation results.

#### 2. We identified a function induction mechanism responsible for this.
* We use mechanistic interpretability technique to understand how model manages to do this. We identified three groups of attention heads, connected in a way similar to the induction head mechanism.
* See `2_circuit_discovery/gemma_2_9b.ipynb` for a demo of circuit discovery and result visualization.

#### 3. Function induction heads work in parallel and each emits a fraction of the +1 function.
* See `3_circuit_eval_and_analysis/function_vector.ipynb` for a demo of function vector style analysis.

#### 4. Function induction enables broader task generalization.


