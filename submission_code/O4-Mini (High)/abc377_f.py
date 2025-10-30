import sys
from bisect import bisect_left, bisect_right

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    rows_set = set()
    cols_set = set()
    sum_set = set()
    diff_set = set()
    for _ in range(M):
        a = int(next(it)); b = int(next(it))
        rows_set.add(a)
        cols_set.add(b)
        sum_set.add(a + b)
        diff_set.add(a - b)
    # Convert to lists for iteration
    rows = list(rows_set)
    cols = list(cols_set)
    sums = list(sum_set)
    diffs = list(diff_set)
    R = len(rows)
    C = len(cols)
    S = len(sums)
    D = len(diffs)

    # 1) sum1: sum of sizes of each attacked line
    attacked_rows_total = R * N
    attacked_cols_total = C * N
    attacked_sum_diags_total = 0
    # anti-diagonals i+j = s
    # length = s-1 if s <= N+1 else 2*N+1 - s
    Np1 = N + 1
    twoN1 = 2 * N + 1
    for s in sums:
        if s <= Np1:
            attacked_sum_diags_total += (s - 1)
        else:
            attacked_sum_diags_total += (twoN1 - s)
    attacked_diff_diags_total = 0
    # main-diagonals i-j = d
    # length = N - d if d>=0 else N + d
    for d in diffs:
        if d >= 0:
            attacked_diff_diags_total += (N - d)
        else:
            attacked_diff_diags_total += (N + d)

    sum1 = (attacked_rows_total + attacked_cols_total +
            attacked_sum_diags_total + attacked_diff_diags_total)

    # 2) sum2: subtract intersections of every pair of attacked lines
    sum2 = 0
    # rows x columns: always R*C intersections
    sum2 += R * C

    # prepare sorted lists for bisect
    sums_sorted = sorted(sums)
    diffs_sorted = sorted(diffs)
    twoN = 2 * N

    # rows x anti-diagonals
    tmp = 0
    for r in rows:
        lo = r + 1
        hi = r + N
        # count s in [lo, hi]
        lpos = bisect_left(sums_sorted, lo)
        rpos = bisect_right(sums_sorted, hi)
        tmp += (rpos - lpos)
    sum2 += tmp

    # rows x main-diagonals
    tmp = 0
    for r in rows:
        lo = r - N
        hi = r - 1
        lpos = bisect_left(diffs_sorted, lo)
        rpos = bisect_right(diffs_sorted, hi)
        tmp += (rpos - lpos)
    sum2 += tmp

    # columns x anti-diagonals
    tmp = 0
    for c in cols:
        lo = c + 1
        hi = c + N
        lpos = bisect_left(sums_sorted, lo)
        rpos = bisect_right(sums_sorted, hi)
        tmp += (rpos - lpos)
    sum2 += tmp

    # columns x main-diagonals
    tmp = 0
    for c in cols:
        lo = 1 - c
        hi = N - c
        lpos = bisect_left(diffs_sorted, lo)
        rpos = bisect_right(diffs_sorted, hi)
        tmp += (rpos - lpos)
    sum2 += tmp

    # anti-diagonals x main-diagonals
    # need s+d even, and 1 <= (s+d)/2 <= N, 1 <= (s-d)/2 <= N
    # we split diffs by parity for faster bisect
    diffs_even = [d for d in diffs_sorted if d % 2 == 0]
    diffs_odd  = [d for d in diffs_sorted if d % 2 != 0]
    tmp = 0
    for s in sums:
        # choose diffs of same parity
        if (s & 1) == 0:
            DL = diffs_even
        else:
            DL = diffs_odd
        if not DL:
            continue
        # bounds from 1 <= (s+d)/2 <= N => 2 <= s+d <= 2N => d in [2-s, 2N-s]
        #        1 <= (s-d)/2 <= N => 2 <= s-d <= 2N => d in [s-2N, s-2]
        lo1 = 2 - s
        hi1 = twoN - s
        lo2 = s - twoN
        hi2 = s - 2
        lo = lo1 if lo1 > lo2 else lo2
        hi = hi1 if hi1 < hi2 else hi2
        if lo <= hi:
            lpos = bisect_left(DL, lo)
            rpos = bisect_right(DL, hi)
            tmp += (rpos - lpos)
    sum2 += tmp

    # 3) sum3: add back intersections of triples of lines
    # families (row, col, diag1), (row, col, diag2),
    # (row, diag1, diag2), (col, diag1, diag2)

    # row+col intersections counts for diag1 and diag2
    rc_s = 0
    rc_d = 0
    rc_sd = 0
    sum_set_lookup = sum_set  # alias
    diff_set_lookup = diff_set
    for r in rows:
        for c in cols:
            s_rc = r + c
            d_rc = r - c
            flag_s = (s_rc in sum_set_lookup)
            flag_d = (d_rc in diff_set_lookup)
            if flag_s:
                rc_s += 1
            if flag_d:
                rc_d += 1
            if flag_s and flag_d:
                rc_sd += 1

    # (row, diag1, diag2)
    r_s_d_count = 0
    for r in rows:
        for s in sums:
            d = 2*r - s
            # j = s - r
            j = s - r
            if 1 <= j <= N and d in diff_set_lookup:
                r_s_d_count += 1

    # (col, diag1, diag2)
    c_s_d_count = 0
    for c in cols:
        for s in sums:
            d = s - 2*c
            # i = s - c
            i = s - c
            if 1 <= i <= N and d in diff_set_lookup:
                c_s_d_count += 1

    sum3 = rc_s + rc_d + r_s_d_count + c_s_d_count
    # 4) sum4: subtract intersections of all four lines
    sum4 = rc_sd

    attacked = sum1 - sum2 + sum3 - sum4
    total = N * N
    safe = total - attacked
    # Output result
    sys.stdout.write(str(safe))

# call main
main()