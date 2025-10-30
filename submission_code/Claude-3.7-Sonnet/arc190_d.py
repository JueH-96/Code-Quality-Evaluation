def matrix_mult(A, B, p):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sum_val = 0
            for k in range(n):
                sum_val = (sum_val + A[i][k] * B[k][j]) % p
            result[i][j] = sum_val
    return result

def matrix_pow(A, p, mod):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1  # Identity matrix
    
    while p > 0:
        if p % 2 == 1:
            result = matrix_mult(result, A, mod)
        A = matrix_mult(A, A, mod)
        p //= 2
    
    return result

def sum_of_matrix_powers(A, p):
    n = len(A)
    zeros = []
    for i in range(n):
        for j in range(n):
            if A[i][j] == 0:
                zeros.append((i, j))
    
    num_zeros = len(zeros)
    result = [[0] * n for _ in range(n)]
    
    def generate_and_compute(idx, B):
        if idx == num_zeros:
            B_pow = matrix_pow(B, p, p)
            for i in range(n):
                for j in range(n):
                    result[i][j] = (result[i][j] + B_pow[i][j]) % p
            return
        
        i, j = zeros[idx]
        for val in range(1, p):
            B[i][j] = val
            generate_and_compute(idx + 1, B)
        
        # Restore the original value for the next recursion
        B[i][j] = 0
    
    # Start the recursion
    B = [row[:] for row in A]
    generate_and_compute(0, B)
    
    return result

def main():
    N, p = map(int, input().split())
    A = []
    for _ in range(N):
        A.append(list(map(int, input().split())))
    
    result = sum_of_matrix_powers(A, p)
    for row in result:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()