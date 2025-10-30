import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    # Define matrix multiplication for 2x2 matrices
    def mat_mult_2x2(A, B):
        result = [[0, 0], [0, 0]]
        result[0][0] = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD
        result[0][1] = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD
        result[1][0] = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD
        result[1][1] = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD
        return result

    # Define matrix exponentiation for 2x2 matrices
    def mat_pow_2x2(M_mat, power):
        result = [[1, 0], [0, 1]] # Identity matrix
        base = M_mat
        while power > 0:
            if power % 2 == 1:
                result = mat_mult_2x2(result, base)
            base = mat_mult_2x2(base, base)
            power //= 2
        return result

    # Initial values for the recurrence:
    # a_1 = 0
    # a_2 = M * (M-1)
    a2_val = (M % MOD * ((M - 1 + MOD) % MOD)) % MOD
    a1_val = 0

    # The transition matrix T for [a_k, a_{k-1}]^T
    # T = [[(M-2), (M-1)], [1, 0]]
    T = [
        [((M - 2 + MOD) % MOD), ((M - 1 + MOD) % MOD)],
        [1, 0]
    ]

    # The starting vector V_2 = [a_2, a_1]^T
    V_start = [[a2_val], [a1_val]] # Column vector

    # If N = 2, we directly use a_2. The matrix exponentiation with N-2=0
    # will correctly return the identity matrix, and multiplying by V_start
    # will yield V_start itself, so final_vec[0][0] will be a_2.
    # This means the logic covers N=2 correctly.
    
    # Calculate T^(N-2)
    # The power is N-2 because V_N = T^(N-2) * V_2
    power_val = N - 2
    
    transformed_T_matrix = mat_pow_2x2(T, power_val)
    
    # Calculate the final vector: result_vector = transformed_T_matrix * V_start
    # Since V_start is a 2x1 matrix (column vector), the result is also a 2x1 matrix.
    # We only need the first element (a_N).
    
    # Manually compute the matrix-vector product for efficiency (2x2 * 2x1)
    final_an = (transformed_T_matrix[0][0] * V_start[0][0] + 
                transformed_T_matrix[0][1] * V_start[1][0]) % MOD
    
    print(final_an)

solve()