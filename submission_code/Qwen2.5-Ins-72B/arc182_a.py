# YOUR CODE HERE
import sys

MOD = 998244353

def solve(N, Q, operations):
    dp = [1] * (N + 1)
    max_val = [0] * (N + 1)
    
    for i in range(Q):
        P, V = operations[i]
        new_dp = [0] * (N + 1)
        new_max_val = [0] * (N + 1)
        
        for j in range(1, N + 1):
            if j <= P:
                if V >= max_val[j]:
                    new_dp[j] = (new_dp[j] + dp[j]) % MOD
                    new_max_val[j] = max(new_max_val[j], V)
            else:
                if V >= max_val[j]:
                    new_dp[j] = (new_dp[j] + dp[j]) % MOD
                    new_max_val[j] = max(new_max_val[j], V)
                if V >= max_val[P]:
                    new_dp[j] = (new_dp[j] + dp[P]) % MOD
                    new_max_val[j] = max(new_max_val[j], V)
        
        dp = new_dp
        max_val = new_max_val
    
    return sum(dp) % MOD

input = sys.stdin.read
data = input().split()
N = int(data[0])
Q = int(data[1])
operations = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(Q)]

print(solve(N, Q, operations))