import sys
from collections import defaultdict

def solve(N, P, A):
    # Create a mapping from index to its permutation index
    perm_map = {i: P[i] - 1 for i in range(N)}
    
    # Create a mapping from index to its value in A
    A_map = {i: A[i] for i in range(N)}
    
    # Create a mapping from permutation index to its original index
    inv_perm_map = {v: k for k, v in perm_map.items()}
    
    # Create a mapping from index to its cycle
    cycle_map = defaultdict(list)
    visited = set()
    
    for i in range(N):
        if i not in visited:
            cycle = []
            while i not in visited:
                visited.add(i)
                cycle.append(i)
                i = perm_map[i]
            for idx in cycle:
                cycle_map[idx] = cycle
    
    # Find the lexicographically smallest A
    result = [0] * N
    for i in range(N):
        cycle = cycle_map[i]
        min_val = min(A[j] for j in cycle)
        min_idx = A.index(min_val)
        for j in range(len(cycle)):
            result[cycle[j]] = A[min_idx]
            min_idx = perm_map[min_idx]
    
    return result

# Read input
N = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

# Solve the problem
result = solve(N, P, A)

# Print the result
print(' '.join(map(str, result)))