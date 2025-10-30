# YOUR CODE HERE
import sys
from itertools import product

def main():
    N, K = map(int, sys.stdin.readline().split())
    R = list(map(int, sys.stdin.readline().split()))
    
    # Generate all possible sequences
    sequences = product(*[range(1, r+1) for r in R])
    
    # Filter sequences where the sum is a multiple of K
    valid_sequences = [seq for seq in sequences if sum(seq) % K == 0]
    
    # Sort the valid sequences lexicographically
    valid_sequences.sort()
    
    # Print the valid sequences
    for seq in valid_sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()