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
        result[i][i] = 1
    
    base = [row[:] for row in A]
    
    while exp > 0:
        if exp % 2 == 1:
            result = matrix_multiply(result, base, p)
        base = matrix_multiply(base, base, p)
        exp //= 2
    
    return result

def solve():
    n, p = map(int, input().split())
    A = []
    for _ in range(n):
        row = list(map(int, input().split()))
        A.append(row)
    
    # Find positions of zeros
    zero_positions = []
    for i in range(n):
        for j in range(n):
            if A[i][j] == 0:
                zero_positions.append((i, j))
    
    k = len(zero_positions)
    
    # Initialize result matrix
    result = [[0] * n for _ in range(n)]
    
    # Generate all possible combinations
    def generate_combinations(idx, current_matrix):
        if idx == k:
            # Compute B^p and add to result
            B_p = matrix_power(current_matrix, p, p)
            for i in range(n):
                for j in range(n):
                    result[i][j] = (result[i][j] + B_p[i][j]) % p
            return
        
        i, j = zero_positions[idx]
        for val in range(1, p):
            current_matrix[i][j] = val
            generate_combinations(idx + 1, current_matrix)
        current_matrix[i][j] = 0
    
    # Create a copy of A for modifications
    current = [row[:] for row in A]
    generate_combinations(0, current)
    
    # Print result
    for i in range(n):
        print(' '.join(map(str, result[i])))

solve()