import sys

# Standard modular arithmetic functions
MOD = 998244353

def mat_mul(A, B):
    """Performs matrix multiplication A * B modulo MOD."""
    R1, C1 = len(A), len(A[0])
    R2, C2 = len(B), len(B[0])
    if C1 != R2:
        raise ValueError("Matrix dimensions mismatch for multiplication")
    
    C = [[0] * C2 for _ in range(R1)]
    for i in range(R1):
        for j in range(C2):
            for k in range(C1):
                C[i][j] = (C[i][k] * B[k][j]) % MOD # Corrected order A[i][k] * B[k][j]
            # Summation is implicit in the +=
    return C

def mat_pow(A, K):
    """Computes matrix A to the power K modulo MOD using binary exponentiation."""
    R, C = len(A), len(A[0])
    if R != C:
        raise ValueError("Matrix must be square for exponentiation")
    
    res = [[0] * R for _ in range(R)]
    for i in range(R):
        res[i][i] = 1 # Identity matrix
    
    A_copy = [row[:] for row in A] # Create a copy to avoid modifying original A
    
    while K > 0:
        if K % 2 == 1:
            res = mat_mul(res, A_copy)
        A_copy = mat_mul(A_copy, A_copy)
        K //= 2
    return res

def mat_sub(A, B):
    """Performs matrix subtraction A - B modulo MOD."""
    R, C = len(A), len(A[0])
    res = [[0] * C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            res[i][j] = (A[i][j] - B[i][j] + MOD) % MOD
    return res

def mat_identity(size):
    """Returns an identity matrix of given size."""
    I = [[0] * size for _ in range(size)]
    for i in range(size):
        I[i][i] = 1
    return I

def modinv(x):
    """Computes modular multiplicative inverse of x modulo MOD."""
    return pow(x, MOD - 2, MOD)

def mat_inverse(A):
    """Computes the inverse of a matrix A modulo MOD using Gaussian elimination."""
    n = len(A)
    if n != len(A[0]):
        raise ValueError("Matrix must be square")

    # Augment A with identity matrix
    augmented_A = [[0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            augmented_A[i][j] = A[i][j]
        augmented_A[i][i + n] = 1

    # Apply Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot_row = i
        while pivot_row < n and augmented_A[pivot_row][i] == 0:
            pivot_row += 1
        if pivot_row == n:
            raise ValueError("Matrix is singular") # Matrix is singular

        augmented_A[i], augmented_A[pivot_row] = augmented_A[pivot_row], augmented_A[i]

        pivot_val = augmented_A[i][i]
        inv_pivot_val = modinv(pivot_val)

        # Scale row i to make pivot 1
        for j in range(2 * n):
            augmented_A[i][j] = (augmented_A[i][j] * inv_pivot_val) % MOD

        # Eliminate other rows
        for row_idx in range(n):
            if row_idx != i:
                factor = augmented_A[row_idx][i]
                for col_idx in range(2 * n):
                    augmented_A[row_idx][col_idx] = (augmented_A[row_idx][col_idx] - factor * augmented_A[i][col_idx] + MOD) % MOD
    
    # Extract inverse matrix
    inv_A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inv_A[i][j] = augmented_A[i][j+n]
    return inv_A

# Primes up to M=16 are: 2, 3, 5, 7, 11, 13
primes_list = [2, 3, 5, 7, 11, 13]

def get_prime_factors(num):
    """Returns a list of exponents for each prime in primes_list for a given num."""
    factors = [0] * len(primes_list)
    temp_num = num
    for i, p in enumerate(primes_list):
        if temp_num == 0: # Should not happen for M>=1
            break
        while temp_num > 0 and temp_num % p == 0:
            factors[i] += 1
            temp_num //= p
    return factors

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Special case for M=1: product is always 1, score is always 1.
    # Sum of scores is N * 1 = N.
    if M == 1:
        print(N % MOD)
        return

    num_primes = len(primes_list)
    
    # Precompute P_S values (coefficients of P(1+x))
    # P_S = sum_{a=1}^M (product of v_{p_i}(a) for p_i in S)
    # These values are indexed by bitmask 's_mask' representing subset S.
    vecP = [0] * (1 << num_primes)
    
    # Precompute prime factorizations for numbers from 1 to M
    prime_factorizations_M = [[]] * (M + 1)
    for i in range(1, M + 1):
        prime_factorizations_M[i] = get_prime_factors(i)

    # Calculate vecP[s_mask] for each s_mask
    for s_mask in range(1 << num_primes):
        current_P_S = 0
        for a in range(1, M + 1):
            prod_v = 1
            for p_idx in range(num_primes):
                # If the p_idx-th prime is in subset S (represented by s_mask)
                if (s_mask >> p_idx) & 1: 
                    prod_v = (prod_v * prime_factorizations_M[a][p_idx])
            current_P_S = (current_P_S + prod_v) % MOD
        vecP[s_mask] = current_P_S
    
    # Construct Transition Matrix T
    # T[s_prime][s] = vecP[s XOR s_prime] if s_prime is a subset of s (s_prime & s == s_prime)
    # T[s_prime][s] = 0 otherwise
    # This matrix facilitates the polynomial multiplication v_{k+1}[S] = sum_{A subset S} v_k[A] * P_{S \setminus A}
    # where v_k is a row vector representing the coefficients of P(1+x)^k.
    mat_size = 1 << num_primes # 2^L
    T = [[0] * mat_size for _ in range(mat_size)]
    
    for s_prime in range(mat_size): # row index (mask for S')
        for s in range(mat_size):     # col index (mask for S)
            # Check if S' is a subset of S
            if (s_prime & s) == s_prime:
                T[s_prime][s] = vecP[s ^ s_prime] # s ^ s_prime computes mask for S \ S'
            # else T[s_prime][s] remains 0

    # Compute sum_{k=1}^N T^k = T * (I - T^N) * (I - T)^-1
    # This formula is valid because M > 1, which implies P_empty = M != 1.
    # Since T is upper-triangular with M on the diagonal (if masks are ordered by popcount/lexicographically),
    # (I-T) will have (1-M) on its diagonal, making it invertible.
    
    I = mat_identity(mat_size)
    
    T_pow_N = mat_pow(T, N)
    
    I_minus_T = mat_sub(I, T)
    I_minus_T_inv = mat_inverse(I_minus_T)
    
    I_minus_T_pow_N = mat_sub(I, T_pow_N)

    # sum_T_k = T @ (I_minus_T_inv @ I_minus_T_pow_N)
    sum_T_k = mat_mul(T, I_minus_T_inv)
    sum_T_k = mat_mul(sum_T_k, I_minus_T_pow_N)

    # Initial state vector v0_row: for an empty sequence (length 0), the product is 1,
    # meaning all prime exponents are 0. The corresponding polynomial term is x^0 (mask 0),
    # with coefficient 1. All other coefficients are 0.
    v0_row = [0] * mat_size
    v0_row[0] = 1 # Coefficient for the empty set/x^0 term

    # The result vector is v0_row @ sum_T_k. This represents the coefficients of the
    # sum of polynomials sum_{k=1}^N P(1+x)^k.
    res_coeffs_row = [0] * mat_size
    for j in range(mat_size): # column index for sum_T_k (corresponding to target S)
        for k in range(mat_size): # row index for sum_T_k (corresponding to initial S')
            res_coeffs_row[j] = (res_coeffs_row[j] + v0_row[k] * sum_T_k[k][j]) % MOD

    # The total sum of scores is the sum of all coefficients in res_coeffs_row.
    # This is because the operation was (prod_p (y_p d/dy_p + 1)) on S_N(y)|_{y=1},
    # which simplifies to sum_{T subseteq V} (prod_{p in T} d/dy_p S_N(y))|_{y=1},
    # and these are precisely the coefficients of S_N(1+x).
    total_score = sum(res_coeffs_row) % MOD
    
    print(total_score)

# Call the solve function
solve()