import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
p = int(data[index])
index += 1
A = []
for i in range(N):
    row = []
    for j in range(N):
        a_ij = int(data[index])
        row.append(a_ij)
        index += 1
    A.append(row)

# Define M
M = [[0 for j in range(N)] for i in range(N)]
for i in range(N):
    for j in range(N):
        if A[i][j] != 0:
            M[i][j] = A[i][j]
        else:
            M[i][j] = p - 1

# Define matrix_mult
def matrix_mult(A, B, mod):
    N_len = len(A)
    C = [[0 for _ in range(N_len)] for _ in range(N_len)]
    for i in range(N_len):
        for j in range(N_len):
            for k in range(N_len):
                C[i][j] += (A[i][k] * B[k][j] % mod)
                C[i][j] %= mod
    return C

# Define matrix_power
def matrix_power(M, exp, mod):
    N_len = len(M)
    res = [[1 if i == j else 0 for j in range(N_len)] for i in range(N_len)]
    base = [row[:] for row in M]
    e = exp
    while e > 0:
        if e % 2 == 1:
            res = matrix_mult(res, base, mod)
        base = matrix_mult(base, base, mod)
        e //= 2
    return res

# Compute M^p mod p
M_p = matrix_power(M, p, p)

# Modify the diagonal where A[i][i] == 0
for i in range(N):
    if A[i][i] == 0:
        M_p[i][i] = 0

# Output the result
for i in range(N):
    for j in range(N):
        if j > 0:
            print(' ', end='')  # Space before each number except the first in the row
        print(M_p[i][j], end='')
    print()  # New line after each row