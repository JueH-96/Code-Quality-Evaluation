import sys

def main():
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    q = int(input_data[0])
    idx = 1  # pointer in input_data
    seq = []
    outputs = []
    for _ in range(q):
        t = input_data[idx]
        idx += 1
        if t == '1':                     # append operation
            x = int(input_data[idx]); idx += 1
            seq.append(x)
        else:                            # query operation (t == '2')
            k = int(input_data[idx]); idx += 1
            outputs.append(str(seq[-k]))
    sys.stdout.write('
'.join(outputs))

if __name__ == "__main__":
    main()