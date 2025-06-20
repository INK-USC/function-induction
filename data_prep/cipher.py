import random
import os
import json
import string

def save_jsonl(datapoints, setting_name, filename, output_dir="../data/cipher"):
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(output_dir)
    if setting_name is not None and not os.path.exists(os.path.join(output_dir, setting_name)):
        os.makedirs(os.path.join(output_dir, setting_name))
    with open(os.path.join(output_dir, setting_name, filename), "w") as fout:
        for line in datapoints:
            fout.write(json.dumps(line)+"\n")

def generate_random_string(length):
    letters = string.ascii_letters[:26]  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choices(letters, k=length))

def shift(s: str, offset: int):
    t = ""
    for c in s:
        if c.isupper():
            t += chr(ord("A") + (ord(c) - ord("A") + offset) % 26)
        elif c.islower():
            t += chr(ord("a") + (ord(c) - ord("a") + offset) % 26)
        else:
            raise Exception(f"{c} in {s} is not A-Z a-z")
    return t

def sample_data(n_shot, n_examples, seed, offset, maxlen):
    random.seed(seed)

    all_example = []
    for i in range(n_examples):
        demos_for_this_example = []
        for j in range(n_shot):
            src_str = generate_random_string(maxlen)
            tar_str = shift(src_str, offset)
            demos_for_this_example.append([src_str, tar_str])

        src_str = generate_random_string(maxlen)
        tar_str = shift(src_str, offset)
        while [src_str, tar_str] in demos_for_this_example:
            src_str = generate_random_string(maxlen)
            tar_str = shift(src_str, offset)

        example = {"demonstration": demos_for_this_example, "inference": [src_str, tar_str]}
        all_example.append(example)

    return all_example

def main():

    setting_name = "normal"
    for maxlen in [1]:
        for offset in range(-12, 14, 1): # offset goes from -12 to 13
            datapoints = sample_data(n_shot=32, n_examples=100, seed=42, offset=offset, maxlen=maxlen)
            filename = "cipher_maxlen{}_offset{}.jsonl".format(maxlen, offset)
            save_jsonl(datapoints, setting_name, filename)
            print("{}:{}:{} finished.".format(setting_name, maxlen, offset))


if __name__ == "__main__":
    main()