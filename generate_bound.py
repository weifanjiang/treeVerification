import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input testing data')
parser.add_argument('-o', '--output', help='output path for features')
parser.add_argument('-e', '--epsilon', help='epsilon', type=float)
args = vars(parser.parse_args())

input = open(args['input'], 'r')
one_data = input.readlines()[0]
fout = open(args['output'], 'w')
fout.write('{\n')

eps = args['epsilon']
change = eps * 0.2

keys = list()
for token in one_data.split(" ")[1:]:
    key, val = token.split(":")
    keys.append(key)

for i in range(len(keys)):
    key = keys[i]
    lo = np.random.uniform(-1 * change, change)
    hi = np.random.uniform(-1 * change, change)
    if i == len(keys) - 1:
        fout.write("  \"{}\": [{}, {}]\n".format(key, eps + lo, eps + hi))
    else:
        fout.write("  \"{}\": [{}, {}],\n".format(key, eps + lo, eps + hi))
fout.write("}")