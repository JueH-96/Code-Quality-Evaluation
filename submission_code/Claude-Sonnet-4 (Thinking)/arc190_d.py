def matrix_multiply(A, B, p):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p
    return C

def matrix_power(A, exp, p):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1  # Identity matrix
    
    base = [row[:] for row in A]  # Copy of A
    
    while exp > 0:
        if exp % 2 == 1:
            result = matrix_multiply(result, base, p)
        base = matrix_multiply(base, base, p)
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

if K == 0:
    # No zeros, just compute A^p
    result = matrix_power(A, p, p)
    for i in range(N):
        print(' '.join(map(str, result[i])))
else:
    # Sum of all B^p
    total_sum = [[0] * N for _ in range(N)]
    
    # Generate all possible assignments for zero positions
    def generate_assignments(index, current_assignment):
        if index == K:
            # Create matrix B from current assignment
            B = [row[:] for row in A]  # Copy A
            for k, (i, j) in enumerate(zero_positions):
                B[i][j] = current_assignment[k]
            
            # Compute B^p
            B_power_p = matrix_power(B, p, p)
            
            # Add to total sum
            for i in range(N):
                for j in range(N):
                    total_sum[i][j] = (total_sum[i][j] + B_power_p[i][j]) % p
            return
        
        # Try all values from 1 to p-1 for the current zero position
        for value in range(1, p):
            current_assignment[index] = value
            generate_assignments(index + 1, current_assignment)
    
    current_assignment = [0] * K
    generate_assignments(0, current_assignment)
    
    # Output the result
    for i in range(N):
        print(' '.join(map(str, total_sum[i])))