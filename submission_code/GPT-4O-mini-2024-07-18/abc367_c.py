from itertools import product
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:2 + N]))
    
    sequences = []
    
    # Generate all possible sequences using product
    for seq in product(*(range(1, r + 1) for r in R)):
        if sum(seq) % K == 0:
            sequences.append(seq)
    
    # Sort sequences lexicographically
    sequences.sort()
    
    # Print the results
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()