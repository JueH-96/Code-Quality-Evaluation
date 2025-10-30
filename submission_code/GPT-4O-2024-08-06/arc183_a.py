import math
from itertools import permutations

def find_good_sequence(N, K):
    # Calculate the total number of good sequences
    total_sequences = math.factorial(N * K) // (math.factorial(K) ** N)
    
    # Find the target index
    target_index = (total_sequences + 1) // 2 - 1
    
    # Generate the base sequence
    base_sequence = []
    for i in range(1, N + 1):
        base_sequence.extend([i] * K)
    
    # Generate permutations in lexicographical order
    seen = set()
    for perm in permutations(base_sequence):
        if perm not in seen:
            seen.add(perm)
            if len(seen) - 1 == target_index:
                return perm

def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, K = map(int, data.split())
    
    result = find_good_sequence(N, K)
    print(' '.join(map(str, result)))