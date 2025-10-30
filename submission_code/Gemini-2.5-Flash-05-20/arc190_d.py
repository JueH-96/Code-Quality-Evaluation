import sys

def solve():
    N, p = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(N):
        A.append(list(map(int, sys.stdin.readline().split())))

    # Matrix multiplication function (C = A * B mod mod_val)
    def multiply(mat1, mat2, mod_val):
        result = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    result[i][j] = (result[i][j] + mat1[i][k] * mat2[k][j]) % mod_val
        return result

    # Matrix exponentiation by squaring (mat^exp mod mod_val)
    def matrix_pow(mat, exp, mod_val):
        result = [[0] * N for _ in range(N)]
        # Initialize result as identity matrix
        for i in range(N):
            result[i][i] = 1

        base = mat
        while exp > 0:
            if exp % 2 == 1:
                result = multiply(result, base, mod_val)
            base = multiply(base, base, mod_val)
            exp //= 2
        return result

    # Step 1: Construct matrix M
    M = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                M[i][j] = p - 1 # Replace 0 with p-1
            else:
                M[i][j] = A[i][j]

    # Step 2: Compute M^p mod p
    # Handle p=0 for matrix_pow (result should be identity), though p >= 1 here.
    # If p=1, matrix_pow(M, 1, p) returns M.
    R = matrix_pow(M, p, p)

    # Step 3: Adjust diagonal elements based on original A matrix
    # This specific adjustment logic only applies for p > 2.
    # For p = 2, the logic implicitly works because (p-1) - R_ii is (1 - R_ii).
    # If A_ii = 0, and R_ii = 1 (from M=all_ones_matrix), then 1-1=0.
    # If A_ii = 0, and R_ii = 0, then 1-0=1.
    # For p=2, M is the all-ones matrix. M^p is all-ones matrix.
    # R_ii would be 1. If A_ii=0, then 1-1=0.
    # But sample 2 output is all 1s.
    # My derived rule for p=2 was simple: M^p where 0s are replaced by 1s.
    # The current general rule seems to work for both!

    # My hypothesis for p=2 was: just M^p where 0s are replaced by 1s.
    # This means NO diagonal adjustment.
    # Let's verify p=2 again with the adjustment:
    # A = [[1,0,0],[0,1,0],[0,0,1]], p=2
    # M = [[1,1,1],[1,1,1],[1,1,1]]
    # R = M^2 = [[1,1,1],[1,1,1],[1,1,1]] mod 2
    # Diagonal adjustment:
    # A_11=1 != 0, so R_11 remains 1.
    # A_22=1 != 0, so R_22 remains 1.
    # A_33=1 != 0, so R_33 remains 1.
    # All A_ii are non-zero, so no diagonal elements are adjusted.
    # So for p=2, the result is indeed M^p without further adjustments.
    # The specific condition for adjustment applies only if p>2 OR if A_ii=0.
    # Let's reformulate: only adjust if p > 2 AND A_ii=0.
    # This is equivalent to: if A_ii=0 AND p > 2, then R_ii = (p-1) - R_ii.
    # Otherwise, R_ii remains as it is.

    if p > 2:
        for i in range(N):
            if A[i][i] == 0:
                R[i][i] = ((p - 1) - R[i][i]) % p
                # Handle potential negative results from (p-1 - R[i][i]) if R[i][i] is large.
                # In Python, % handles negative numbers correctly for positive modulus: (-a) % p = (p-a) % p
                # But it is safer to use: (value % p + p) % p. Here R[i][i] is already %p.
                # So ((p-1) - R[i][i] + p) % p is probably best for consistency.
                # However, (p-1) - R[i][i] will be in range [-(p-1), p-1]. Adding p ensures positive.
                # Example: p=3, R_ii=2, (3-1)-2 = 0.
                # Example: p=3, R_ii=0, (3-1)-0 = 2.
                # Looks fine.

    # Step 4: Print the result matrix R
    for i in range(N):
        sys.stdout.write(" ".join(map(str, R[i])) + "
")

solve()