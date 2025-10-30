import sys
from itertools import product

def main():
    n, k = map(int, sys.stdin.readline().split())
    r_list = list(map(int, sys.stdin.readline().split()))
    
    ranges = [range(1, r + 1) for r in r_list]
    sequences = product(*ranges)
    
    valid_sequences = []
    for seq in sequences:
        if sum(seq) % k == 0:
            valid_sequences.append(seq)
    
    for s in valid_sequences:
        print(' '.join(map(str, s)))

if __name__ == "__main__":
    main()