import sys

# ---------- matrix utilities ----------
def mat_mul(A, B, p):
    """multiply two square matrices modulo p"""
    n = len(A)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ri = res[i]
        for k in range(n):
            a = Ai[k]
            if a == 0:
                continue
            Bk = B[k]
            for j in range(n):
                Ri[j] = (Ri[j] + a * Bk[j]) % p
    return res


def mat_pow(mat, e, p):
    """mat ** e  (square matrix)  modulo p"""
    n = len(mat)
    # identity
    res = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in mat]
    while e:
        if e & 1:
            res = mat_mul(res, base, p)
        base = mat_mul(base, base, p)
        e >>= 1
    return res
# ---------- end of matrix utilities ----------


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    it = iter(sys.stdin.read().strip().split())
    N = int(next(it))
    p = int(next(it))

    A = [[int(next(it)) for _ in range(N)] for _ in range(N)]

    # case p == 2 : every zero can only be replaced with 1, so only one matrix B
    if p == 2:
        B = [[1 if A[i][j] == 0 else  A[i][j] for j in range(N)] for i in range(N)]
        BP = mat_mul(B, B, 2)          # B^2  (since p == 2)
        for row in BP:
            print(*[x % 2 for x in row])
        return

    # -------------- p  >  2 ----------------
    # C : keep original non-zero entries, zeros stay zero
    C = [row[:] for row in A]

    # list of zero positions and their count
    zeros = []
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                zeros.append((i, j))
    K = len(zeros)

    # D = C^p   (matrix power)
    D = mat_pow(C, p, p)

    # S1 : extra contribution coming from variables
    S1 = [[0] * N for _ in range(N)]
    off_diag_needed = (p == 3)              # off-diagonal variables matter only when p == 3
    m2 = (p - 2) % p                        # (p - 2) modulo p

    for r, c in zeros:
        if r == c:                          # diagonal zero
            row_r = C[r]
            # add row r
            for j in range(N):
                S1[r][j] = (S1[r][j] + row_r[j]) % p
            # add column r
            for i in range(N):
                S1[i][r] = (S1[i][r] + C[i][r]) % p
            # adjust diagonal so that total becomes 0 (row + column added 2*Crr)
            S1[r][r] = (S1[r][r] + m2 * C[r][r]) % p
        else:                               # off-diagonal zero
            if off_diag_needed:
                # contribution only for p == 3 :  value C[c][r] added to (r,c)
                S1[r][c] = (S1[r][c] + C[c][r]) % p

    # (-1)^K  modulo p
    factor = 1 if (K % 2 == 0) else (p - 1)

    # final answer : factor * ( D + S1 )  modulo p
    out_lines = []
    for i in range(N):
        row_vals = []
        Di = D[i]
        S1i = S1[i]
        for j in range(N):
            val = (Di[j] + S1i[j]) % p
            val = (val * factor) % p
            row_vals.append(val)
        out_lines.append(' '.join(map(str, row_vals)))

    print('
'.join(out_lines))


if __name__ == "__main__":
    main()