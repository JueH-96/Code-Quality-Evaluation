def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    R_rows = set()
    C_cols = set()
    S_s = set()
    D_diff = set()

    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        R_rows.add(a)
        C_cols.add(b)
        S_s.add(a + b)
        D_diff.add(a - b)

    R = len(R_rows) * N
    C = len(C_cols) * N

    sum_s = 0
    for s in S_s:
        if 2 <= s <= 2 * N:
            if s <= N + 1:
                sum_s += s - 1
            else:
                sum_s += 2 * N + 1 - s

    sum_d = 0
    for d in D_diff:
        abs_d = abs(d)
        count = N - abs_d
        sum_d += count if count > 0 else 0

    rc = len(R_rows) * len(C_cols)

    rs = 0
    for r in R_rows:
        for s in S_s:
            if 1 <= s - r <= N:
                rs += 1

    rd = 0
    for r in R_rows:
        for d in D_diff:
            j = r - d
            if 1 <= j <= N:
                rd += 1

    cs = 0
    for c in C_cols:
        for s in S_s:
            i = s - c
            if 1 <= i <= N:
                cs += 1

    cd = 0
    for c in C_cols:
        for d in D_diff:
            i = c + d
            if 1 <= i <= N:
                cd += 1

    sd = 0
    for s in S_s:
        for d in D_diff:
            if (s + d) % 2 == 0:
                i = (s + d) // 2
                j = (s - d) // 2
                if 1 <= i <= N and 1 <= j <= N:
                    sd += 1

    pairwise = rs + rd + cs + cd + sd + rc

    rcs = 0
    for r in R_rows:
        for c in C_cols:
            if (r + c) in S_s:
                rcs += 1

    rcd = 0
    for r in R_rows:
        for c in C_cols:
            if (r - c) in D_diff:
                rcd += 1

    rstd = 0
    for r in R_rows:
        for s in S_s:
            j = s - r
            if 1 <= j <= N:
                d_candidate = r - j
                if d_candidate in D_diff:
                    rstd += 1

    cSD = 0
    for c in C_cols:
        for s in S_s:
            i = s - c
            if 1 <= i <= N:
                d_candidate = i - c
                if d_candidate in D_diff:
                    cSD += 1

    triple = rcs + rcd + rstd + cSD

    four = 0
    for r in R_rows:
        for c in C_cols:
            if (r + c) in S_s and (r - c) in D_diff:
                four += 1

    total_forbidden = R + C + sum_s + sum_d - pairwise + triple - four
    answer = N * N - total_forbidden
    print(answer)

if __name__ == "__main__":
    main()