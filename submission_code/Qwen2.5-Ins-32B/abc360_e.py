import sys

def mod_inverse(a, m=998244353):
    return pow(a, m-2, m)

def solve(N, K):
    mod = 998244353
    # Initial probability of black ball being at position 1
    prob = [1] + [0] * (N-1)
    
    # Matrix for transition probabilities
    matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i][j] = (N-1)**2 % mod
            else:
                matrix[i][j] = (N-1) * 2 % mod
        matrix[i][i] = (matrix[i][i] + 1) % mod
    
    # Matrix exponentiation to find the probability after K operations
    result = [0] * N
    result[0] = 1
    for _ in range(K):
        new_result = [0] * N
        for i in range(N):
            for j in range(N):
                new_result[i] = (new_result[i] + result[j] * matrix[j][i]) % mod
        result = new_result
    
    # Calculate the expected value
    expected_value = sum((i+1) * result[i] for i in range(N)) % mod
    return expected_value

# Read input
N, K = map(int, sys.stdin.readline().split())
# Solve the problem
answer = solve(N, K)
# Write the answer to stdout
print(answer)