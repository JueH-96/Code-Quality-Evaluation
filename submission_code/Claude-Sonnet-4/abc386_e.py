from itertools import combinations

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

max_xor = 0

# Generate all combinations of K elements
for combo in combinations(range(N), K):
    # Calculate XOR of selected elements
    current_xor = 0
    for i in combo:
        current_xor ^= A[i]
    
    # Update maximum XOR
    max_xor = max(max_xor, current_xor)

print(max_xor)