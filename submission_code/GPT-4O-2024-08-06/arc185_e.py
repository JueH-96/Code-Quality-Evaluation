import sys
import math
from collections import defaultdict

MOD = 998244353

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # dp[i][g] will store the number of subsequences ending at i with gcd g
    dp = [defaultdict(int) for _ in range(N)]
    
    # Result array
    results = [0] * N
    
    for i in range(N):
        current = A[i]
        # Start a new subsequence with A[i]
        dp[i][current] += 1
        
        # Update dp[i] based on previous dp[j] where j < i
        for j in range(i):
            for g, count in dp[j].items():
                new_gcd = math.gcd(g, current)
                dp[i][new_gcd] = (dp[i][new_gcd] + count) % MOD
        
        # Calculate the score for all subsequences ending at i
        score = 0
        for g, count in dp[i].items():
            score = (score + g * count) % MOD
        
        # Add this score to the result for m = i + 1
        if i > 0:
            results[i] = (results[i - 1] + score) % MOD
        else:
            results[i] = score
    
    # Print results
    for result in results:
        print(result)