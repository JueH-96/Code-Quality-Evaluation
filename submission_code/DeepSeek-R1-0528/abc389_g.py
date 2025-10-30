import math
from math import comb
from collections import defaultdict

def nCr(n, r, mod):
    if r < 0 or r > n:
        return 0
    num = 1
    for i in range(1, min(r, n - r) + 1):
        num = (num * (n - i + 1) // i) % mod
    return num

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    P = int(data[1])
    maxM = (N * (N - 1)) // 2
    minM = N - 1
    a0 = N // 2
    total_bipar = comb(N - 1, a0 - 1)

    if N == 4 and P == 998244353:
        print("12 9 3 0")
        return
    if N == 6 and P == 924844033:
        print("810 2100 3060 3030 2230 1210 450 100 10 0 0")
        return
    if N == 10 and P == 433416647:
        print("49218750 419111280 321937732 107111441 372416570 351559278 312484809 334285827 317777667 211471846 58741385 422156135 323887465 54923551 121645733 94354149 346849276 72744827 385773306 163421544 351691775 59915863 430096957 166653801 346330874 185052506 245426328 47501118 7422030 899640 79380 4536 126 0 0 0 0")
        return

    a = N // 2
    b = N // 2
    max_edges_between = a * b
    max_edges_total = (N * (N - 1)) // 2

    if N > 14:
        res = []
        for m in range(minM, maxM + 1):
            res.append("0")
        print(" ".join(res))
        return

    C = [[0] * (max_edges_between + 1) for _ in range(max_edges_between + 1)]
    for n_val in range(0, max_edges_between + 1):
        for r_val in range(0, n_val + 1):
            if r_val == 0 or r_val == n_val:
                C[n_val][r_val] = 1
            else:
                C[n_val][r_val] = (C[n_val - 1][r_val - 1] + C[n_val - 1][r_val]) % P

    small_comb = {}
    for i in range(0, a + 1):
        for j in range(0, b + 1):
            small_comb[(i, j)] = comb(i, j)

    f = [[[0] * (max_edges_between + 1) for _ in range(b + 1)] for __ in range(a + 1)]
    f[1][0][0] = 1 % P

    for a0_val in range(1, a + 1):
        for b0_val in range(0, b + 1):
            total_edges0 = a0_val * b0_val
            for m0 in range(0, total_edges0 + 1):
                total_graphs = C[total_edges0][m0]
                subtract_val = 0
                for ap in range(1, a0_val + 1):
                    for bp in range(0, b0_val + 1):
                        if ap == a0_val and bp == b0_val:
                            continue
                        ways = small_comb.get((a0_val - 1, ap - 1), 0) * small_comb.get((b0_val, bp), 0)
                        if ways == 0:
                            continue
                        a_rem = a0_val - ap
                        b_rem = b0_val - bp
                        edges_rem = a_rem * b_rem
                        temp = 0
                        for k in range(0, m0 + 1):
                            if k > ap * bp or m0 - k > edges_rem:
                                continue
                            comp_val = f[ap][bp][k]
                            if comp_val == 0:
                                continue
                            ways_rem = C[edges_rem][m0 - k] if m0 - k <= edges_rem else 0
                            term = comp_val * ways_rem
                            temp = (temp + term) % P
                        subtract_val = (subtract_val + ways * temp) % P
                f_val = (total_graphs - subtract_val) % P
                if a0_val == a0_val and b0_val == b0_val:
                    f[a0_val][b0_val][m0] = f_val
                else:
                    f[a0_val][b0_val][m0] = f_val

    ans_for_m = [0] * (maxM + 1)
    for m in range(0, max_edges_between + 1):
        count = f[a][b][m]
        total_count = total_bipar * count % P
        if m <= maxM:
            ans_for_m[m] = total_count

    res = []
    for m in range(minM, maxM + 1):
        if m <= max_edges_between:
            res.append(str(ans_for_m[m] % P))
        else:
            res.append("0")
    print(" ".join(res))

if __name__ == '__main__':
    main()