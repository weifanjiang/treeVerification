import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Output average box')
    parser.add_argument('-i', '--input', type=str, help='path of output file from verification algorithm', required=True)
    return parser.parse_args()

def main(args):
    with open(args.input, "r") as f:
        lines = f.readlines()
        start = len(lines) - 1
        while "display average box:" not in lines[start]:
            start -= 1
        end = start + 1
        while "}" not in lines[end]:
            end += 1
        box = lines[start + 1:end]
        box = sorted(box, key=lambda x: int(x.strip(" {\n").split(":")[0]))
        
        print('{')
        for line in box:
            print(line.replace("\n", "").replace("{", " "))
        print('}')

if __name__=='__main__':
    args = parse_args()
    main(args)
