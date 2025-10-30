import sys
from itertools import product

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))

    sequences = product(*[range(1, r + 1) for r in R])
    valid_sequences = [seq for seq in sequences if sum(seq) % K == 0]

    for seq in sorted(valid_sequences):
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()