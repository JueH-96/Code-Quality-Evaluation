# YOUR CODE HERE
MOD = 998244353

def matrix_multiply(A, B):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % MOD
    return result

def matrix_power(A, k):
    n = len(A)
    if k == 0:
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    if k == 1:
        return A
    half = matrix_power(A, k // 2)
    result = matrix_multiply(half, half)
    if k % 2:
        result = matrix_multiply(result, A)
    return result

N, M, K = map(int, input().split())

adj_matrix = [[0] * N for _ in range(N)]
for i in range(N):
    adj_matrix[i][(i + 1) % N] = 1

for _ in range(M):
    x, y = map(int, input().split())
    adj_matrix[x - 1][y - 1] = 1

result_matrix = matrix_power(adj_matrix, K)
print(result_matrix[0][0])