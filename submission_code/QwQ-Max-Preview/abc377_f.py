def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    R = set()
    C = set()
    S = set()
    D = set()

    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        R.add(a)
        C.add(b)
        S.add(a + b)
        D.add(a - b)

    # Compute sum_S
    sum_S = 0
    for s in S:
        if 2 <= s <= 2 * N:
            sum_S += min(s - 1, 2 * N + 1 - s)

    # Compute sum_D
    sum_D = 0
    for d in D:
        if abs(d) < N:
            sum_D += (N - abs(d))

    r = len(R)
    c = len(C)

    # Compute pairwise intersections
    pair_rc = r * c

    pair_rs = 0
    for a in R:
        for s in S:
            j = s - a
            if 1 <= j <= N:
                pair_rs += 1

    pair_rd = 0
    for a in R:
        for d in D:
            j = a - d
            if 1 <= j <= N:
                pair_rd += 1

    pair_cs = 0
    for b in C:
        for s in S:
            i = s - b
            if 1 <= i <= N:
                pair_cs += 1

    pair_cd = 0
    for b in C:
        for d in D:
            i = d + b
            if 1 <= i <= N:
                pair_cd += 1

    pair_sd = 0
    for s in S:
        for d in D:
            if (s + d) % 2 == 0:
                i = (s + d) // 2
                j = (s - d) // 2
                if 1 <= i <= N and 1 <= j <= N:
                    pair_sd += 1

    # Compute triple intersections
    triple_rcs = 0
    for a in R:
        for b in C:
            if (a + b) in S:
                triple_rcs += 1

    triple_rcd = 0
    for a in R:
        for b in C:
            if (a - b) in D:
                triple_rcd += 1

    triple_rsd = 0
    for s in S:
        for d_val in D:
            if (s + d_val) % 2 == 0:
                a = (s + d_val) // 2
                j = (s - d_val) // 2
                if 1 <= a <= N and 1 <= j <= N and a in R:
                    triple_rsd += 1

    triple_csd = 0
    for b in C:
        for s in S:
            i = s - b
            if 1 <= i <= N:
                if (s - 2 * b) in D:
                    triple_csd += 1

    # Compute four-way intersection
    four_way = 0
    for a in R:
        for b in C:
            if (a + b) in S and (a - b) in D:
                four_way += 1

    # Compute U using inclusion-exclusion formula
    U = (r * N + c * N + sum_S + sum_D) 
    U -= (pair_rc + pair_rs + pair_rd + pair_cs + pair_cd + pair_sd)
    U += (triple_rcs + triple_rcd + triple_rsd + triple_csd)
    U -= four_way

    answer = N * N - U
    print(answer)

if __name__ == '__main__':
    main()