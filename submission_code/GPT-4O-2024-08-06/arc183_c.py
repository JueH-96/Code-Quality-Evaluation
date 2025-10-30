# YOUR CODE HERE
import sys
from itertools import combinations

MOD = 998244353

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    conditions = []
    index = 2
    for _ in range(M):
        L = int(data[index]) - 1
        R = int(data[index + 1]) - 1
        X = int(data[index + 2]) - 1
        conditions.append((L, R, X))
        index += 3
    
    # dp[mask] will store the number of valid permutations for the subset of numbers represented by mask
    dp = [0] * (1 << N)
    dp[0] = 1  # Base case: one way to arrange an empty set
    
    for mask in range(1 << N):
        count = bin(mask).count('1')
        if count == N:
            continue
        
        for i in range(N):
            if not (mask & (1 << i)):  # If i is not in the current mask
                new_mask = mask | (1 << i)
                valid = True
                
                # Check all conditions
                for L, R, X in conditions:
                    if L <= count <= R and count == X:
                        if i == max(range(N), key=lambda k: (mask & (1 << k)) == 0 and L <= k <= R):
                            valid = False
                            break
                
                if valid:
                    dp[new_mask] = (dp[new_mask] + dp[mask]) % MOD
    
    print(dp[(1 << N) - 1])