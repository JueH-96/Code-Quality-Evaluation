def compute_forbidden(N, M, pieces):
    R = set()
    C = set()
    D1 = set()
    D2 = set()

    for a, b in pieces:
        R.add(a)
        C.add(b)
        s = a + b
        D1.add(s)
        d = a - b
        D2.add(d)

    # Calculate individual set sizes
    a = len(R) * N
    b = len(C) * N

    sum_d1_len = 0
    for s in D1:
        if s <= N + 1:
            sum_d1_len += s - 1
        else:
            sum_d1_len += (2 * N + 1 - s)
    c = sum_d1_len

    sum_d2_len = 0
    for d in D2:
        if abs(d) < N:
            sum_d2_len += (N - abs(d))
    d = sum_d2_len

    # Pairwise intersections
    sum_rc = len(R) * len(C)

    sum_rd1 = 0
    for r in R:
        for s in D1:
            j = s - r
            if 1 <= j <= N:
                sum_rd1 += 1

    sum_rd2 = 0
    for r in R:
        for d_ in D2:
            i = r - d_
            if 1 <= i <= N:
                sum_rd2 += 1

    sum_cd1 = 0
    for c_ in C:
        for s in D1:
            j = s - c_
            if 1 <= j <= N:
                sum_cd1 += 1

    sum_cd2 = 0
    for c_ in C:
        for d_ in D2:
            i = c_ + d_
            if 1 <= i <= N:
                sum_cd2 += 1

    sum_d1d2 = 0
    for s in D1:
        for d_ in D2:
            if (s + d_) % 2 == 0:
                i = (s + d_) // 2
                j = (s - d_) // 2
                if 1 <= i <= N and 1 <= j <= N:
                    sum_d1d2 += 1

    # Triple intersections
    sum_rcd1 = 0
    for r in R:
        for c_ in C:
            s = r + c_
            if s in D1:
                sum_rcd1 += 1

    sum_rcd2 = 0
    for r in R:
        for c_ in C:
            d_ = r - c_
            if d_ in D2:
                sum_rcd2 += 1

    sum_r_d1_d2 = 0
    for r in R:
        for s in D1:
            d_ = 2 * r - s
            if d_ in D2:
                j = s - r
                if 1 <= j <= N:
                    sum_r_d1_d2 += 1

    sum_c_d1_d2 = 0
    for c_ in C:
        for s in D1:
            d_ = s - 2 * c_
            if d_ in D2:
                i = s - c_
                if 1 <= i <= N:
                    sum_c_d1_d2 += 1

    # Four-way intersection
    four_way = 0
    for r in R:
        for c_ in C:
            s = r + c_
            d_ = r - c_
            if s in D1 and d_ in D2:
                four_way += 1

    # Inclusion-exclusion formula
    forbidden = a + b + c + d
    forbidden -= (sum_rc + sum_rd1 + sum_rd2 + sum_cd1 + sum_cd2 + sum_d1d2)
    forbidden += (sum_rcd1 + sum_rcd2 + sum_r_d1_d2 + sum_c_d1_d2)
    forbidden -= four_way

    return forbidden

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    pieces = []
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        pieces.append((a, b))
    forbidden = compute_forbidden(N, M, pieces)
    available = N * N - forbidden
    print(available)

if __name__ == "__main__":
    main()