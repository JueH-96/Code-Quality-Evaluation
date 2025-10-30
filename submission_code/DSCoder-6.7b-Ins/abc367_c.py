from itertools import product, combinations

def solve(n, k, r):
    # Generate all possible sequences of length n with elements between 1 and r
    sequences = list(product(range(1, r+1), repeat=n))
    
    # Filter sequences where the sum of elements is a multiple of k
    sequences = [seq for seq in sequences if sum(seq) % k == 0]
    
    # Sort sequences in lexicographical order
    sequences.sort()
    
    # Print sequences
    for seq in sequences:
        print(' '.join(map(str, seq)))

# Read inputs from stdin
n, k = map(int, input().split())
r = list(map(int, input().split()))

# Solve the problem
solve(n, k, r)