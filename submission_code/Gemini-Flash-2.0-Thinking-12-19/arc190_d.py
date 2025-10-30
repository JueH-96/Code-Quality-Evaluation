# Matrix multiplication
def multiply_matrices(mat1, mat2, n, p):
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] = (result[i][j] + mat1[i][k] * mat2[k][j]) % p
    return result

# Matrix exponentiation by squaring
def matrix_pow(matrix, power, n, p):
    if power == 0:
        identity = [[0] * n for _ in range(n)]
        for i in range(n):
            identity[i][i] = 1
        return identity
    if power == 1:
        # Ensure entries are within [0, p-1)
        return [[matrix[i][j] % p for j in range(n)] for i in range(n)]
        
    if power % 2 == 1:
        return multiply_matrices(matrix, matrix_pow(matrix, power - 1, n, p), n, p)
    else:
        half_pow = matrix_pow(matrix, power // 2, n, p)
        return multiply_matrices(half_pow, half_pow, n, p)

# Read input
N, p = map(int, input().split())
A = []
zero_count = 0
for _ in range(N):
    row = list(map(int, input().split()))
    for x in row:
        if x == 0:
            zero_count += 1
    A.append(row)

if p == 2:
    # For p=2, there is only one B matrix where zeros are replaced by 1s.
    # This B matrix is obtained by replacing all zeros in A with 1s.
    # For p=2, the set {1, ..., p-1} is just {1}.
    # So B_ij = 1 if A_ij = 0, and B_ij = A_ij if A_ij = 1.
    # This means B_ij = 1 for all i,j. B is the all-ones matrix J.
    # The sum is just B^p = J^2 mod 2.
    # (J^2)_ij = sum_k J_ik J_kj = sum_k 1*1 = N.
    # J^2 = N * J.
    # Result is N * J mod 2.
    result = [[(N % 2) for _ in range(N)] for _ in range(N)]
else:
    # For p > 2, the sum S = sum_B B^p mod p.
    # Let A' be A with zeros kept as zeros. A'_ij = A_ij if A_ij != 0, else 0.
    # The matrix B is obtained by replacing A_ij=0 with v_ij in {1, ..., p-1}.
    # Based on analysis using sums of powers and paths, the sum can be calculated as:
    # S = (A')^p + T mod p
    # Where T is a correction matrix. The T matrix accounts for contributions
    # from paths of length p that use p-1 traversals of a diagonal zero entry (i0, i0)
    # and one traversal of a non-zero entry (a,b).
    # These paths are of two types: a -> i0 -> ... -> i0 (from a to i0) and
    # i0 -> ... -> i0 -> b (from i0 to b).
    # Path a -> i0 requires A[a][i0] != 0 and A[i0][i0] == 0. It contributes -A[a][i0] to S[a][i0].
    # Path i0 -> b requires A[i0][b] != 0 and A[i0][i0] == 0. It contributes -A[i0][b] to S[i0][b].

    # Let A_prime be A with zeros kept as zeros
    A_prime = [[A[i][j] if A[i][j] != 0 else 0 for j in range(N)] for i in range(N)]

    # Compute (A')^p mod p
    A_prime_p = matrix_pow(A_prime, p, N, p)

    # Calculate T matrix
    T = [[0] * N for _ in range(N)]

    # Iterate over all possible indices i0 (0 to N-1)
    for i0 in range(N):
        # Check if A[i0][i0] is a zero entry in the original matrix A
        if A[i0][i0] == 0:
            # Add contributions for paths ending at i0 (type a -> i0 -> ... -> i0)
            # These paths contribute to matrix entries in column i0.
            # For each row r, if A[r][i0] is non-zero in the original matrix A,
            # add -A[r][i0] to T[r][i0].
            for r in range(N):
                 if A[r][i0] != 0:
                     T[r][i0] = (T[r][i0] - A[r][i0]) % p

            # Add contributions for paths starting from i0 (type i0 -> ... -> i0 -> b)
            # These paths contribute to matrix entries in row i0.
            # For each column c, if A[i0][c] is non-zero in the original matrix A,
            # add -A[i0][c] to T[i0][c].
            for c in range(N):
                if A[i0][c] != 0:
                    T[i0][c] = (T[i0][c] - A[i0][c]) % p

    # Sum the matrices: S = (A')^p + T mod p
    result = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            result[i][j] = (A_prime_p[i][j] + T[i][j]) % p
            # Ensure positive modulo
            if result[i][j] < 0:
                result[i][j] += p

# Print result
for row in result:
    print(*row)