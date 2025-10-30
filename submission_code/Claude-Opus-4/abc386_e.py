# YOUR CODE HERE
from itertools import combinations

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    max_xor = 0
    
    # Generate all combinations of K elements
    for combo in combinations(range(N), K):
        xor_val = 0
        for idx in combo:
            xor_val ^= A[idx]
        max_xor = max(max_xor, xor_val)
    
    print(max_xor)

solve()