import sys
from math import gcd
from functools import lru_cache
from collections import defaultdict
input = sys.stdin.read

MOD = 998244353

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Precompute all possible gcds for pairs in A to avoid recomputation
    max_A = max(A)
    gcd_cache = {}
    
    def get_gcd(x, y):
        if (x, y) in gcd_cache:
            return gcd_cache[(x, y)]
        if (y, x) in gcd_cache:
            return gcd_cache[(y, x)]
        result = gcd(x, y)
        gcd_cache[(x, y)] = result
        return result
    
    # To store results for each m
    results = [0] * N
    
    # We need to calculate for each m = 1 to N
    for m in range(1, N + 1):
        # Calculate the sum of scores of all subsequences of A[0:m]
        current_A = A[:m]
        subsequence_count = 1 << m  # 2^m
        
        # We will use dynamic programming to count the contribution of each gcd pair
        # dp[i][g] will store the number of subsequences ending at index i with gcd so far as g
        dp = [defaultdict(int) for _ in range(m)]
        
        # Initialize dp for single element subsequences
        for i in range(m):
            dp[i][current_A[i]] = 1
        
        # Fill dp for subsequences longer than 1
        for i in range(m):
            for j in range(i + 1, m):
                new_gcd = get_gcd(current_A[i], current_A[j])
                dp[j][new_gcd] += dp[i][current_A[i]]
                dp[j][new_gcd] %= MOD
        
        # Now calculate the sum of scores using the dp table
        total_score = 0
        for i in range(1, m):
            for g, count in dp[i].items():
                total_score += g * count
                total_score %= MOD
        
        results[m-1] = total_score
    
    # Output results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()