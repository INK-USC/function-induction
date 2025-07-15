import random
import os
import json

def generate_one_example(cmin, cmax, nmin, nmax):
    """generate one addition example (a+b=c), where a,b \in [nmin, nmax] and c \in [cmin, cmax]. 
    rejection sampling is used."""
    a, b = -1e10, -1e10
    while (a+b > cmax) or (a+b < cmin):
        a = random.randint(nmin, nmax)
        b = random.randint(nmin, nmax)
    return a, b

def sample_data_normal(n_shot, n_examples, cmin, cmax, nmax):
    all_examples = []
    for i in range(n_examples):
        test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax)

        demos_for_this_example = []
        while len(demos_for_this_example) < n_shot:
            a, b = generate_one_example(cmin, cmax, nmin=0, nmax=nmax)
            while ([a, b] == [test_a, test_b]):
                # make sure the test example is differet from ICL examples
                a, b = generate_one_example(cmin, cmax, nmin=0, nmax=nmax) 
            
            demos_for_this_example.append([a,b])
        
        example = {"demonstration": demos_for_this_example, "inference": [test_a, test_b]}
        all_examples.append(example)
    return all_examples

def sample_data_setting1(n_shot, n_examples, cmin, cmax, nmax):
    all_examples = []
    for i in range(n_examples):
        test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax)

        demos_for_this_example = []
        while len(demos_for_this_example) < n_shot:
            a, b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax)
            while (a + b == test_a + test_b):
                # make sure a+b != test_c
                a, b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax) 
            
            demos_for_this_example.append([a,b])
        
        example = {"demonstration": demos_for_this_example, "inference": [test_a, test_b]}
        all_examples.append(example)
    return all_examples

def sample_data_setting2(n_shot, n_examples, cmin, cmax, nmax):
    all_examples = []

    for i in range(n_examples):

        demos_for_this_example = []
        all_cs = []
        while len(demos_for_this_example) < n_shot:
            a, b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax)
            demos_for_this_example.append([a,b])
            all_cs.append(a+b)

        test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax)
        while (test_a+test_b not in all_cs) or ([test_a, test_b] in demos_for_this_example):
            # make sure test_c appears in c's
            test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=0, nmax=nmax) 

        example = {"demonstration": demos_for_this_example, "inference": [test_a, test_b]}
        all_examples.append(example)
    return all_examples

def sample_data(n_shot, n_examples, seed, offset, nmax, setting_name):
    """generate `n_examples` examples of one digit addition;
    it has `n_shot` in-context examples and one test example
    """
    random.seed(seed)

    nmin = {9: 0, 99: 10, 999: 100}[nmax]
    cmin = max(nmin, nmin-offset)
    cmax = min(nmax, nmax-offset)

    sampling_func_dict = {"normal": sample_data_normal, "setting1": sample_data_setting1, "setting2": sample_data_setting2}
    all_examples = sampling_func_dict[setting_name](n_shot, n_examples, cmin, cmax, nmax)

    return all_examples

def save_jsonl(datapoints, setting_name, filename, output_dir="../data/addition"):
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(output_dir)
    if setting_name is not None and not os.path.exists(os.path.join(output_dir, setting_name)):
        os.makedirs(os.path.join(output_dir, setting_name))
    with open(os.path.join(output_dir, setting_name, filename), "w") as fout:
        for line in datapoints:
            fout.write(json.dumps(line)+"\n")

def main():
    skip_func = lambda nmax, offset: (nmax == 9 and abs(offset) > 2)

    setting_name = "normal"
    for nmax in [9, 99, 999]:
        for offset in range(-10, 11, 1): # offset goes from -10 to 10
            if skip_func(nmax, offset):
                continue
            datapoints = sample_data(n_shot=32, n_examples=100, seed=42, offset=offset, nmax=nmax, setting_name=setting_name)
            filename = "addition_nmax{}_offset{}.jsonl".format(nmax, offset)
            save_jsonl(datapoints, setting_name, filename)
            print("{}:{}:{} finished.".format(setting_name, nmax, offset))

    setting_name = "setting1"
    for nmax in [9, 99, 999]:
        for offset in range(-10, 11, 1):
            if skip_func(nmax, offset):
                continue
            datapoints = sample_data(n_shot=32, n_examples=100, seed=42, offset=offset, nmax=nmax, setting_name=setting_name)
            filename = "addition_nmax{}_offset{}.jsonl".format(nmax, offset)
            save_jsonl(datapoints, setting_name, filename)
            print("{}:{}:{} finished.".format(setting_name, nmax, offset))

    setting_name = "setting2"
    for nmax in [9, 99, 999]:
        for offset in range(-10, 11, 1):
            if skip_func(nmax, offset):
                continue
            datapoints = sample_data(n_shot=32, n_examples=100, seed=42, offset=offset, nmax=nmax, setting_name=setting_name)
            filename = "addition_nmax{}_offset{}.jsonl".format(nmax, offset)
            save_jsonl(datapoints, setting_name, filename)
            print("{}:{}:{} finished.".format(setting_name, nmax, offset))

if __name__ == "__main__":
    main()