import sys

def matrix_mul(A, B, p):
    N = len(A)
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p
    return C

def matrix_pow(A, power, p):
    N = len(A)
    if power == 0:
        I = [[0] * N for _ in range(N)]
        for i in range(N):
            I[i][i] = 1
        return I
    if power == 1:
        # Input elements are already modulo p
        return A
    
    if power % 2 == 0:
        half_pow = matrix_pow(A, power // 2, p)
        return matrix_mul(half_pow, half_pow, p)
    else:
        half_pow = matrix_pow(A, power // 2, p)
        # A elements are already modulo p
        return matrix_mul(A, matrix_mul(half_pow, half_pow, p), p)

# Read input
N, p = map(int, sys.stdin.readline().split())
A = []
for _ in range(N):
    A.append(list(map(int, sys.stdin.readline().split())))

# Case p=2
if p == 2:
    result = [[N % 2] * N for _ in range(N)]
else:
    # Case p > 2
    # Create A_ones matrix
    A_ones = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                A_ones[i][j] = 1
            else:
                # Elements are already 0 <= A_ij <= p-1
                A_ones[i][j] = A[i][j]

    # Compute A_ones^p mod p
    result = matrix_pow(A_ones, p, p)

# Print output
for row in result:
    print(*row)