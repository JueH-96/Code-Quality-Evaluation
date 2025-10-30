# YOUR CODE HERE
import sys
import math
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    dp = defaultdict(int)
    result = []
    
    for i in range(N):
        new_dp = defaultdict(int)
        new_dp[A[i]] += 1
        
        for g, count in dp.items():
            new_g = math.gcd(g, A[i])
            new_dp[new_g] += count
        
        dp = new_dp
        
        score = 0
        for g, count in dp.items():
            score = (score + g * count) % MOD
        
        result.append(score)
    
    for res in result:
        print(res)

if __name__ == "__main__":
    main()