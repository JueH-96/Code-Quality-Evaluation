def matrix_mult(A, B, p):
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
    
    base = [row[:] for row in A]  # Copy A
    
    while exp > 0:
        if exp % 2 == 1:
            result = matrix_mult(result, base, p)
        base = matrix_mult(base, base, p)
        exp //= 2
    
    return result

def solve():
    N, p = map(int, input().split())
    A = []
    for _ in range(N):
        row = list(map(int, input().split()))
        A.append(row)
    
    # Find positions of zeros
    zero_positions = []
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                zero_positions.append((i, j))
    
    K = len(zero_positions)
    
    # Initialize result matrix
    result = [[0] * N for _ in range(N)]
    
    # Generate all possible combinations
    def generate_combinations(pos, current_matrix):
        if pos == K:
            # Compute current_matrix^p and add to result
            powered = matrix_power(current_matrix, p, p)
            for i in range(N):
                for j in range(N):
                    result[i][j] = (result[i][j] + powered[i][j]) % p
            return
        
        i, j = zero_positions[pos]
        for val in range(1, p):
            current_matrix[i][j] = val
            generate_combinations(pos + 1, current_matrix)
            current_matrix[i][j] = 0  # Reset for next iteration
    
    # Create initial matrix
    B = [row[:] for row in A]  # Copy A
    generate_combinations(0, B)
    
    # Print result
    for i in range(N):
        print(' '.join(map(str, result[i])))

solve()