# YOUR CODE HERE
MOD = 998244353

def matrix_mult(A, B, mod):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % mod
    return result

def matrix_pow(matrix, power, mod):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = matrix
    while power:
        if power % 2 == 1:
            result = matrix_mult(result, base, mod)
        base = matrix_mult(base, base, mod)
        power //= 2
    return result

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    K = int(input[index + 2])
    index += 3

    adj_matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        adj_matrix[i][(i + 1) % N] = 1

    for _ in range(M):
        X = int(input[index]) - 1
        Y = int(input[index + 1]) - 1
        adj_matrix[X][Y] = 1
        index += 2

    result_matrix = matrix_pow(adj_matrix, K, MOD)
    print(result_matrix[0][0])

if __name__ == "__main__":
    main()