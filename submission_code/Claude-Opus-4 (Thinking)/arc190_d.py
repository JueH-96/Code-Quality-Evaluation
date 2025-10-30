def matrix_mult(A, B, p):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p
    return C

def matrix_pow(A, exp, p):
    n = len(A)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in A]
    
    while exp > 0:
        if exp % 2 == 1:
            result = matrix_mult(result, base, p)
        base = matrix_mult(base, base, p)
        exp //= 2
    
    return result

N, p = map(int, input().split())
A = []
for _ in range(N):
    row = list(map(int, input().split()))
    A.append(row)

# Find zero positions
zero_positions = []
for i in range(N):
    for j in range(N):
        if A[i][j] == 0:
            zero_positions.append((i, j))

K = len(zero_positions)
total = [[0] * N for _ in range(N)]

# Enumerate all possible B matrices
for mask in range((p-1) ** K):
    B = [row[:] for row in A]
    temp = mask
    for i, j in zero_positions:
        B[i][j] = (temp % (p-1)) + 1
        temp //= (p-1)
    
    # Compute B^p
    B_pow_p = matrix_pow(B, p, p)
    
    # Add to total
    for i in range(N):
        for j in range(N):
            total[i][j] = (total[i][j] + B_pow_p[i][j]) % p

# Print result
for i in range(N):
    print(' '.join(map(str, total[i])))