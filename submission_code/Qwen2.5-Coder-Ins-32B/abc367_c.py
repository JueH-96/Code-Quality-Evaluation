# YOUR CODE HERE
import sys
from itertools import product

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    R = list(map(int, input[2:]))

    sequences = []
    for seq in product(*(range(1, r+1) for r in R)):
        if sum(seq) % K == 0:
            sequences.append(seq)

    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()