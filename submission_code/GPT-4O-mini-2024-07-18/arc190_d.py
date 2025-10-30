def matrix_power(mat, exp, mod):
    """Computes matrix `mat` raised to the power `exp` modulo `mod`."""
    n = len(mat)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Identity matrix
    while exp:
        if exp % 2 == 1:
            result = matrix_mult(result, mat, mod)
        mat = matrix_mult(mat, mat, mod)
        exp //= 2
    return result

def matrix_mult(A, B, mod):
    """Multiplies two matrices A and B under modulo mod."""
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n)) % mod
    return result

def sum_of_powers(N, p, A):
    # Count the number of zeros in A
    K = sum(1 for i in range(N) for j in range(N) if A[i][j] == 0)
    
    # Prepare the sum matrix
    sum_matrix = [[0] * N for _ in range(N)]
    
    # Iterate through all possible values for the zeros
    from itertools import product
    
    # Generate all combinations of values for the zeros
    for values in product(range(1, p), repeat=K):
        B = [row[:] for row in A]  # Copy A to B
        idx = 0
        
        # Fill B with the current combination of values
        for i in range(N):
            for j in range(N):
                if B[i][j] == 0:
                    B[i][j] = values[idx]
                    idx += 1
        
        # Compute B^p
        B_powered = matrix_power(B, p, p)
        
        # Add B^p to the sum_matrix
        for i in range(N):
            for j in range(N):
                sum_matrix[i][j] = (sum_matrix[i][j] + B_powered[i][j]) % p
    
    return sum_matrix

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and p
    N, p = map(int, data[0].split())
    
    # Read matrix A
    A = [list(map(int, line.split())) for line in data[1:N+1]]
    
    # Calculate the sum of B^p
    result = sum_of_powers(N, p, A)
    
    # Print the result
    for row in result:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()