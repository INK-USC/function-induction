import random
import os
import json

def to_base(n, base):
    if n == 0:
        return '0'
    digits = []
    while n:
        digits.append(str(n % base))
        n //= base
    return ''.join(digits[::-1])

def generate_one_example(cmin, cmax, nmin, nmax, base, ndigit):
    """generate one addition example (a+b=c), where a,b \in [nmin, nmax] and c \in [cmin, cmax]. 
    rejection sampling is used."""
    a, b = -1e10, -1e10
    # or (int(to_base(a, base)) + int(to_base(b, base)) == int(to_base(a+b, base)))
    while (a+b > cmax) or (a+b < cmin) \
        or len(to_base(a, base)) != ndigit or len(to_base(b, base)) != ndigit or len(to_base(a+b, base)) != ndigit:
        a = random.randint(nmin, nmax)
        b = random.randint(nmin, nmax)
    return a, b

def sample_data(n_shot, n_examples, setting_name, base, cmin, cmax, ndigit):
    all_examples = []
    for i in range(n_examples):

        # normal: no intervention
        # setting 1: base-k and base-10 addition results are the same
        # setting 2: base-k and base-10 addition results are different (e.g., involves +2 and carry over in base-8)
        if setting_name == "normal":
            test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
        elif setting_name == "setting1":
            test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
            while int(to_base(test_a, base)) % 10 + int(to_base(test_b, base)) % 10 >= base:
                test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
        elif setting_name == "setting2":
            test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
            c0 = int(to_base(test_a, base)) % 10 + int(to_base(test_b, base)) % 10
            while c0 >= 10 or c0 < base:
                test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
                c0 = int(to_base(test_a, base)) % 10 + int(to_base(test_b, base)) % 10
        elif setting_name == "setting3":
            test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
            c0 = int(to_base(test_a, base)) % 10 + int(to_base(test_b, base)) % 10
            while c0 < 10:
                test_a, test_b = generate_one_example(cmin=cmin, cmax=cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
                c0 = int(to_base(test_a, base)) % 10 + int(to_base(test_b, base)) % 10

        demos_for_this_example = []
        while len(demos_for_this_example) < n_shot:
            a, b = generate_one_example(cmin, cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit)
            while ([a, b] == [test_a, test_b]):
                # make sure the test example is differet from ICL examples
                a, b = generate_one_example(cmin, cmax, nmin=cmin, nmax=cmax, base=base, ndigit=ndigit) 
            
            demos_for_this_example.append([a,b])
        
        example = {"demonstration": demos_for_this_example, "inference": [test_a, test_b]}
        all_examples.append(example)

    return all_examples

def save_jsonl(datapoints, setting_name, filename, output_dir="../data/addition_base"):
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(output_dir)
    if setting_name is not None and not os.path.exists(os.path.join(output_dir, setting_name)):
        os.makedirs(os.path.join(output_dir, setting_name))
    with open(os.path.join(output_dir, setting_name, filename), "w") as fout:
        for line in datapoints:
            fout.write(json.dumps(line)+"\n")

def main():

    seed = 42

    for setting_name in ["normal", "setting1", "setting2", "setting3"]:
        for ndigit in [2, 3]:
            for base in range(6, 10, 1): # offset goes from -10 to 10

                # if setting_name != "normal" and base < 8: continue

                random.seed(seed)

                cmin = max(base ** (ndigit-1), 10 ** (ndigit-1))
                cmax = base ** ndigit - 1

                datapoints = sample_data(n_shot=32, n_examples=100, setting_name=setting_name, base=base, 
                                        cmin=cmin, cmax=cmax, ndigit=ndigit)
                # print(datapoints[:4])
                filename = "addition_ndigits{}_base{}.jsonl".format(ndigit, base)
                save_jsonl(datapoints, setting_name, filename)
                print("{}:{}:{} finished.".format(setting_name, ndigit, base))

if __name__ == "__main__":
    main()