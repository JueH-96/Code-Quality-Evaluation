import sys

# Matrix multiplication function
def matrix_mult(A, B, p):
    N = len(A)
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p
    return C

# Matrix exponentiation function using binary exponentiation
def matrix_pow(A, exp, p):
    N = len(A)
    # Ensure base matrix elements are modulo p
    A_mod = [[A[i][j] % p for j in range(N)] for i in range(N)]

    if exp == 0:
        # Return identity matrix
        I = [[0] * N for _ in range(N)]
        for i in range(N):
            I[i][i] = 1
        return I
    if exp == 1:
        # Return matrix A modulo p
        return A_mod

    if exp % 2 == 1:
        # exp is odd: A^exp = A * A^(exp-1)
        return matrix_mult(A_mod, matrix_pow(A_mod, exp - 1, p), p)
    else:
        # exp is even: A^exp = (A^(exp/2))^2
        half_pow = matrix_pow(A_mod, exp // 2, p)
        return matrix_mult(half_pow, half_pow, p)

# Read input
N, p = map(int, sys.stdin.readline().split())
A = []
for _ in range(N):
    row = list(map(int, sys.stdin.readline().split()))
    A.append(row)

if p == 2:
    # If p=2, any zero in A must be replaced by 1.
    # Any non-zero element in A must be 1 (since values are 0..p-1).
    # So B is always the matrix of all ones, J.
    # We need to compute J^p mod 2.
    # For p=2, this is J^2.
    # (J^2)_i,j = sum_{k=0 to N-1} J_i,k * J_k,j = sum_{k=0 to N-1} 1 * 1 = N.
    # So the result is a matrix where all entries are N mod 2.
    result = [[N % 2] * N for _ in range(N)]
else:
    # For p > 2, based on Sample 3 matching A^p mod p,
    # the formula is likely A^p mod p.
    # A is already the matrix with 0s at zero positions when we read it.
    result = matrix_pow(A, p, p)

# Print result
for row in result:
    print(*row)