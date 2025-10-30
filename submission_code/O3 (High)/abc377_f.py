import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    # attacked lines
    rows = set()
    cols = set()
    diag_sum = set()   # i + j   (anti–diagonal, slope -1)
    diag_diff = set()  # i - j   (main     diagonal, slope +1)

    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        rows.add(a)
        cols.add(b)
        diag_sum.add(a + b)
        diag_diff.add(a - b)

    R = list(rows)
    C = list(cols)
    S = list(diag_sum)
    D = list(diag_diff)

    nR = len(R)
    nC = len(C)

    # ---------- single sets ----------
    A = nR * N                              # rows
    B = nC * N                              # columns

    # diagonals with constant i + j  (values from 2 .. 2N)
    C_size = 0
    twoN_plus1 = 2 * N + 1
    for s in S:
        if s <= N + 1:
            C_size += s - 1
        else:
            C_size += twoN_plus1 - s

    # diagonals with constant i - j (values from -(N-1) .. N-1)
    D_size = 0
    for t in D:
        D_size += N - abs(t)

    # ---------- pair intersections ----------
    AB = nR * nC

    # helper sets for fast membership
    S_set = set(S)
    D_set = set(D)
    C_set = set(C)  # for possible membership tests later if wanted

    # Row & (i + j)
    AC = 0
    for r in R:
        for s in S:
            c = s - r
            if 1 <= c <= N:
                AC += 1

    # Row & (i - j)
    AD = 0
    for r in R:
        for t in D:
            c = r - t
            if 1 <= c <= N:
                AD += 1

    # Column & (i + j)
    BC = 0
    for c in C:
        for s in S:
            r = s - c
            if 1 <= r <= N:
                BC += 1

    # Column & (i - j)
    BD = 0
    for c in C:
        for t in D:
            r = t + c
            if 1 <= r <= N:
                BD += 1

    # (i + j) & (i - j)
    CD = 0
    for s in S:
        for t in D:
            if (s + t) & 1:          # parity check, must be even
                continue
            r = (s + t) // 2
            c = (s - t) // 2
            if 1 <= r <= N and 1 <= c <= N:
                CD += 1

    # ---------- triple intersections (rows & columns always single square) ----------
    ABC = ABD = ABCD = 0
    for r in R:
        for c in C:
            s = r + c
            t = r - c
            inS = s in S_set
            inD = t in D_set
            if inS:
                ABC += 1
            if inD:
                ABD += 1
            if inS and inD:
                ABCD += 1

    # Row & (i + j) & (i - j)
    ACD = 0
    for r in R:
        for s in S:
            c = s - r
            if 1 <= c <= N:
                t = r - c      # 2r - s
                if t in D_set:
                    ACD += 1

    # Column & (i + j) & (i - j)
    BCD = 0
    for c in C:
        for s in S:
            r = s - c
            if 1 <= r <= N:
                t = r - c      # s - 2c
                if t in D_set:
                    BCD += 1

    # ---------- inclusion–exclusion ----------
    singles = A + B + C_size + D_size
    pairs   = AB + AC + AD + BC + BD + CD
    triples = ABC + ABD + ACD + BCD
    quadruple = ABCD

    attacked = singles - pairs + triples - quadruple

    total_squares = N * N
    safe = total_squares - attacked

    print(safe)

if __name__ == "__main__":
    main()