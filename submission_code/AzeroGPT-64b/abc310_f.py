from collections import deque

MOD = 998244353

def matmul(A, B, mod):
    """Multiply two matrices A and B with mod operation"""
    n, m = len(A), len(A[0])
    p = len(B[0])
    res = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                res[i][j] += A[i][k] * B[k][j]
            res[i][j] %= mod
    return res

def matrix_power(M, power, mod):
    """Raise a square matrix to the power and perform mod operation"""
    n = len(M)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Identity matrix
    while power > 0:
        if power % 2 == 1:
            result = matmul(result, M, mod)
        M = matmul(M, M, mod)
        power //= 2
    return result

def multiply_vectors(vector, matrix, mod):
    """Multiply a vector by a matrix with mod operation"""
    result = [0] * len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[j] += vector[i] * matrix[i][j]
            result[j] %= mod
    return result

def solve_10(dices):
    W = [1, 10, 9, 10, 9, 9, 9, 9, 9, 9, 1]
    C = [[1] * 11 for _ in range(len(dices))]
    for i in range(len(dices)):
        for j in range(1, min(11, dices[i]) + 1):
            C[i][j] = C[i][j - 1] * (dices[i] - j + 1) * pow(j, MOD - 2, MOD) % MOD

    I = matrix_power(C, 2, MOD)

    dp_memo = deque([[1]])
    ans = 0
    for d in dices:
        new_dp_memo = []
        for dp in dp_memo:
            cur = multiply_vectors(dp, I, MOD)
            new_dp_memo.append(cur)
            for i in range(1, min(11, d) + 1):
                cur[i:] = [cur[j] * C[-1][i] * W[j] % MOD for j in range(i, 11)]
            new_dp_memo.append(cur)
        dp_memo = new_dp_memo
    for dp in dp_memo:
        ans += dp[10]
        ans %= MOD
    return ans

def solve(diques):
    S = sum(diques)
    cnt_11 = S // 11
    A = [min(11, d) for d in diques]
    A = [a - 1 for a in A]
    for i in range(len(A)):
        if A[i] == -1:
            A[i] = 10
            cnt_11 -= 1
    ans = quickpow(11, S % 11 + 11 * cnt_11, MOD)
    ans = ans * solve_10(A) % MOD
    return ans

def inv(x):
    return pow(x, -1, MOD)

def quickpow(base, n, mod):
    ans = 1
    while n:
        if n & 1:
            ans = ans * base % mod
        base = base * base % mod
        n >>= 1
    return ans

if __name__ == "__main__":
    N = int(input())
    A = list(map(int, input().split()))

    if sum(A) < 10:
        result = 0
    else:
        if 10 in A:
            result = quickpow(10, sum([min(a, 9) for a in A]) + 1, MOD)
        else:
            result = solve(A)

    print(result)