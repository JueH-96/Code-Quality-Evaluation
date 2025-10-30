# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
    except StopIteration:
        return

    # For each piece we read (a,b). A piece attacks its row, its column,
    # and all squares with constant (r+c) and constant (r-c).
    attacked_rows = set()
    attacked_cols = set()
    diagSet = set()   # all possible r+c values attacked
    antiSet = set()   # all possible r-c values attacked
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        attacked_rows.add(a)
        attacked_cols.add(b)
        diagSet.add(a + b)
        antiSet.add(a - b)

    # Only candidate squares are those in safe rows and safe columns.
    # Represent the safe set A = {r: 1<= r <= N and r not in attacked_rows}
    # as union of intervals.
    def get_intervals(N, attacked):
        arr = sorted(attacked)
        intervals = []
        start = 1
        for r in arr:
            if r > start:
                intervals.append((start, r - 1))
            start = r + 1
        if start <= N:
            intervals.append((start, N))
        return intervals

    safe_row_intervals = get_intervals(N, attacked_rows)
    safe_col_intervals = get_intervals(N, attacked_cols)
    # Count number of safe rows and safe cols.
    safe_rows_count = 0
    for l, r in safe_row_intervals:
        safe_rows_count += (r - l + 1)
    safe_cols_count = 0
    for l, r in safe_col_intervals:
        safe_cols_count += (r - l + 1)
    base = safe_rows_count * safe_cols_count

    # Given a candidate sum d and safe intervals for rows and cols,
    # count the number of pairs (r, c) in A×B with r+c = d.
    def conv_sum(d, intervalsR, intervalsC):
        tot = 0
        for (rL, rR) in intervalsR:
            for (cL, cR) in intervalsC:
                # A candidate (r,c) satisfies r+c = d if and only if
                # r is in I and c = d - r lies in J.
                # Equivalently, r must lie in I ∩ [d - cR, d - cL].
                L = max(rL, d - cR)
                R = min(rR, d - cL)
                if L <= R:
                    tot += (R - L + 1)
        return tot

    # Similarly, count the number of pairs (r, c) in A×B with r - c = t.
    def conv_diff(t, intervalsR, intervalsC):
        tot = 0
        for (rL, rR) in intervalsR:
            for (cL, cR) in intervalsC:
                # We need r in I and (r - t) in J.
                L = max(rL, cL + t)
                R = min(rR, cR + t)
                if L <= R:
                    tot += (R - L + 1)
        return tot

    # We need a helper to check membership in a union of intervals.
    def in_intervals(x, intervals):
        lo = 0
        hi = len(intervals)
        while lo < hi:
            mid = (lo + hi) // 2
            a, b = intervals[mid]
            if x < a:
                hi = mid
            elif x > b:
                lo = mid + 1
            else:
                return True
        return False

    # Determine the effective range for candidate sums and differences.
    if safe_row_intervals:
        safe_r_min = safe_row_intervals[0][0]
        safe_r_max = safe_row_intervals[-1][1]
    else:
        safe_r_min, safe_r_max = 10**18, -10**18
    if safe_col_intervals:
        safe_c_min = safe_col_intervals[0][0]
        safe_c_max = safe_col_intervals[-1][1]
    else:
        safe_c_min, safe_c_max = 10**18, -10**18
    sum_min = safe_r_min + safe_c_min
    sum_max = safe_r_max + safe_c_max
    diff_min = safe_r_min - safe_c_max
    diff_max = safe_r_max - safe_c_min

    # We only care about those attacked diagonal sums/differences that can actually occur.
    diag_list = [d for d in diagSet if d >= sum_min and d <= sum_max]
    anti_list = [t for t in antiSet if t >= diff_min and t <= diff_max]

    # Compute the total count (over A×B) of attacked candidates by sum and by diff.
    sumCount = 0
    for d in diag_list:
        sumCount += conv_sum(d, safe_row_intervals, safe_col_intervals)
    diffCount = 0
    for t in anti_list:
        diffCount += conv_diff(t, safe_row_intervals, safe_col_intervals)

    # The intersection: (r,c) in A×B for which r+c = d and r-c = t.
    # Note that if r+c = d and r-c = t then r = (d+t)//2, c = (d-t)//2 provided d+t is even.
    interCount = 0
    for d in diag_list:
        for t in anti_list:
            # only consider if d+t is even
            if (d + t) & 1:
                continue
            r = (d + t) >> 1
            c = (d - t) >> 1
            # Check (r,c) is in safe candidates.
            if in_intervals(r, safe_row_intervals) and in_intervals(c, safe_col_intervals):
                interCount += 1

    # By inclusion–exclusion the total attacked in A×B due to diagonals is:
    attacked_diag = sumCount + diffCount - interCount

    # The answer is then:
    answer = base - attacked_diag

    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()