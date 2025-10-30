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

def matrix_power(A, n):
    if n == 1:
        return A
    if n % 2 == 0:
        half = matrix_power(A, n // 2)
        return matrix_multiply(half, half)
    else:
        return matrix_multiply(A, matrix_power(A, n - 1))

N, M = map(int, input().split())

if M == 1:
    print(0)
else:
    transition = [[M - 1 if i != j else 0 for j in range(M)] for i in range(M)]
    result_matrix = matrix_power(transition, N)
    total = sum(result_matrix[i][j] for i in range(M) for j in range(M)) % MOD
    print(total)