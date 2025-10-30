# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % MOD
    return result

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    conditions = []
    index = 2
    for _ in range(M):
        L = int(data[index])
        R = int(data[index + 1])
        X = int(data[index + 2])
        conditions.append((L, R, X))
        index += 3
    
    # Initialize dp array
    dp = [0] * (N + 1)
    dp[0] = 1
    
    # Process each condition
    for L, R, X in conditions:
        new_dp = dp[:]
        for i in range(L, R + 1):
            if i != X:
                new_dp[i] = (new_dp[i] + dp[i - 1]) % MOD
        dp = new_dp
    
    # Calculate the number of valid permutations
    result = factorial(N)
    for i in range(1, N + 1):
        result = (result * dp[i]) % MOD
    
    print(result)

solve()