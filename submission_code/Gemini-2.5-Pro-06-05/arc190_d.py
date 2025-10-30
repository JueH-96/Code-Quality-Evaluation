import sys

def mat_mul(A, B, mod):
    """Multiplies two NxN matrices A and B modulo mod."""
    N = len(A)
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j])
            C[i][j] %= mod
    return C

def mat_pow(A, n, mod):
    """Computes A^n modulo mod for an NxN matrix A."""
    N = len(A)
    res = [[0] * N for _ in range(N)]
    for i in range(N):
        res[i][i] = 1
    
    base = A
    while n > 0:
        if n % 2 == 1:
            res = mat_mul(res, base, mod)
        base = mat_mul(base, base, mod)
        n //= 2
    return res

def solve():
    """Main function to solve the problem."""
    try:
        input = sys.stdin.readline
        N_str, p_str = input().split()
        N, p = int(N_str), int(p_str)
        A = [list(map(int, input().split())) for _ in range(N)]
    except (IOError, ValueError):
        # Handle empty input for local testing
        return

    if p == 2:
        val = N % 2
        for _ in range(N):
            print(*([val] * N))
        return

    K = 0
    zeros = []
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                K += 1
                zeros.append((i, j))
    
    A_p = mat_pow(A, p, p)
    
    C_corr = [[0] * N for _ in range(N)]
    
    for r, c in zeros:
        if r == c:  # Diagonal zero
            # Contribution is A E_rr + E_rr A
            for i in range(N):
                C_corr[i][r] = (C_corr[i][r] + A[i][r]) % p
            for j in range(N):
                C_corr[r][j] = (C_corr[r][j] + A[r][j]) % p
            # A[r][r] is 0, so its double addition to C_corr[r][r] is 0.
        elif p == 3:  # Off-diagonal zero, only matters for p=3
            # Contribution is A_cr * E_rc
            C_corr[r][c] = (C_corr[r][c] + A[c][r]) % p

    result = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            result[i][j] = (A_p[i][j] + C_corr[i][j]) % p

    if K % 2 == 1:
        for i in range(N):
            for j in range(N):
                result[i][j] = (p - result[i][j]) % p

    for row in result:
        print(*row)

if __name__ == "__main__":
    solve()