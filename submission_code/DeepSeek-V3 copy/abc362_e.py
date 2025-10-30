MOD = 998244353

def count_arithmetic_subsequences(N, A):
    if N == 1:
        return [1]
    
    dp = [{} for _ in range(N)]
    total = [0] * (N + 1)
    
    for i in range(N):
        dp[i][1] = 1
        total[1] += 1
    
    for i in range(1, N):
        for j in range(i):
            diff = A[i] - A[j]
            for length in dp[j]:
                new_length = length + 1
                if new_length > N:
                    continue
                if new_length not in dp[i]:
                    dp[i][new_length] = 0
                dp[i][new_length] = (dp[i][new_length] + dp[j][length]) % MOD
                total[new_length] = (total[new_length] + dp[j][length]) % MOD
    
    result = []
    for k in range(1, N+1):
        result.append(total[k] % MOD)
    return result

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute the result
result = count_arithmetic_subsequences(N, A)

# Print the result
print(' '.join(map(str, result)))