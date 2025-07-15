import json
import os

def process_dataset(datapoints, n_icl_examples, offset):
    new_datapoints = []

    for dp in datapoints:
        normal_input = []
        contrast_input = []

        for j in range(n_icl_examples):
            a, b = dp["demonstration"][j]
            normal_input.append("{}+{}={}".format(a,b,a+b))
            contrast_input.append("{}+{}={}".format(a,b,a+b+offset))
            
        normal_input = "\n".join(normal_input)
        contrast_input = "\n".join(contrast_input)

        if n_icl_examples > 0:
            normal_input += "\n"
            contrast_input += "\n"

        a, b = dp["inference"]
        normal_input += "{}+{}=".format(a, b)
        normal_output = str(a+b)
        contrast_input += "{}+{}=".format(a, b)
        contrast_output = str(a+b+offset)
        
        new_datapoints.append({
            "normal_input": normal_input, "normal_output": normal_output,
            "contrast_input": contrast_input, "contrast_output": contrast_output
        })
    
    return new_datapoints

def read_jsonl(filename):
    with open(filename) as fin:
        lines = fin.readlines()
    json_lines = [json.loads(line) for line in lines]
    return json_lines

def save_jsonl(datapoints, filename,output_dir, setting_name):
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(output_dir)
    if setting_name is not None and not os.path.exists(os.path.join(output_dir, setting_name)):
        os.makedirs(os.path.join(output_dir, setting_name))
    with open(os.path.join(output_dir, setting_name, filename), "w") as fout:
        for line in datapoints:
            fout.write(json.dumps(line)+"\n")