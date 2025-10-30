def matrix_multiply(A, B, N, p):
    result = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % p
    return result

def matrix_power(A, power, N, p):
    if power == 0:
        result = [[1 if i == j else 0 for j in range(N)] for i in range(N)]
        return result
    if power == 1:
        return [[A[i][j] % p for j in range(N)] for i in range(N)]
    
    half = matrix_power(A, power // 2, N, p)
    if power % 2 == 0:
        return matrix_multiply(half, half, N, p)
    else:
        temp = matrix_multiply(half, half, N, p)
        return matrix_multiply(temp, A, N, p)

def add_matrices(A, B, N, p):
    result = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            result[i][j] = (A[i][j] + B[i][j]) % p
    return result

# Read input
N, p = map(int, input().split())
A = []
zero_positions = []
for i in range(N):
    row = list(map(int, input().split()))
    A.append(row)
    for j in range(N):
        if row[j] == 0:
            zero_positions.append((i, j))

# If there are no zeros, just compute A^p
if not zero_positions:
    result = matrix_power(A, p, N, p)
    for row in result:
        print(*row)
else:
    # Initialize result matrix with zeros
    result = [[0] * N for _ in range(N)]
    
    # For each possible combination of non-zero values at zero positions
    for mask in range((p-1) ** len(zero_positions)):
        # Create matrix B by filling zeros with values 1 to p-1
        B = [row[:] for row in A]
        temp = mask
        for idx, (i, j) in enumerate(zero_positions):
            value = (temp % (p-1)) + 1  # Convert to 1..p-1
            B[i][j] = value
            temp //= (p-1)
        
        # Compute B^p and add to result
        powered = matrix_power(B, p, N, p)
        result = add_matrices(result, powered, N, p)

    # Print result
    for row in result:
        print(*row)