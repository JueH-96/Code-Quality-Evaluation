# YOUR CODE HERE
import sys
from itertools import product

def main():
    # Read input
    input = sys.stdin.read().strip().split()
    N = int(input[0])
    K = int(input[1])
    R = list(map(int, input[2:2+N]))
    
    # Generate all possible sequences
    ranges = [range(1, r_i + 1) for r_i in R]
    all_sequences = product(*ranges)
    
    # Filter sequences where the sum is a multiple of K
    valid_sequences = [seq for seq in all_sequences if sum(seq) % K == 0]
    
    # Sort sequences in lexicographical order (product already does this)
    valid_sequences.sort()
    
    # Print the valid sequences
    for seq in valid_sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()