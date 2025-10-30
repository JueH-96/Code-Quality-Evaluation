import sys
input = sys.stdin.read
from math import gcd
from functools import lru_cache

MOD = 998244353

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # To store all possible values for each S_i
    possible_values = [set() for _ in range(N)]
    
    # Initial possible values for S_1
    possible_values[0].add(1)  # Start with 1 to scale later
    
    # Calculate possible values for each S_i based on A_i-1
    for i in range(1, N):
        prev_values = possible_values[i-1]
        a = A[i-1]
        current_values = possible_values[i]
        
        # We need to find all pairs (p, q) such that p * q = a
        for p in range(1, int(a**0.5) + 1):
            if a % p == 0:
                q = a // p
                # p and q are such that p * q = a
                # S_i can be p or q times any of the previous S_{i-1}
                for v in prev_values:
                    current_values.add(p * v)
                    current_values.add(q * v)
    
    # Now we have all possible values for each S_i, but we need to ensure gcd(S_1, ..., S_N) = 1
    # We will use dynamic programming to calculate the sum of all products of good sequences
    # dp[i][g] will represent the sum of products of sequences up to S_i with gcd of all elements = g
    dp = [{} for _ in range(N)]
    
    # Initialize dp for S_1
    for v in possible_values[0]:
        dp[0][v] = v
    
    # Fill dp for subsequent S_i
    for i in range(1, N):
        for v in possible_values[i]:
            for g, sum_prod in dp[i-1].items():
                new_g = gcd(g, v)
                if new_g in dp[i]:
                    dp[i][new_g] = (dp[i][new_g] + sum_prod * v) % MOD
                else:
                    dp[i][new_g] = (sum_prod * v) % MOD
    
    # We are interested in dp[N-1][1], which is the sum of products of all good sequences
    result = dp[N-1].get(1, 0)  # Get the sum for gcd = 1, if not present, return 0
    print(result)