# YOUR CODE HERE
def max_function_composition(N, K, functions):
    dp = [1] * (N + 1)
    
    for _ in range(K):
        new_dp = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j:
                    a, b = functions[i]
                    new_dp[i] = max(new_dp[i], a * dp[j] + b)
        dp = new_dp
    
    return max(dp)

N, K = map(int, input().split())
functions = [0] + [tuple(map(int, input().split())) for _ in range(N)]

result = max_function_composition(N, K, functions)
print(result)