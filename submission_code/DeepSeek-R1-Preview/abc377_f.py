import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    rows = set()
    columns = set()
    diag_sum = set()
    diag_diff = set()

    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        rows.add(a)
        columns.add(b)
        diag_sum.add(a + b)
        diag_diff.add(a - b)

    # Convert to sorted lists for binary search
    rows_sorted = sorted(rows)
    columns_sorted = sorted(columns)
    columns_set = columns  # using set for O(1) lookups

    r_count = len(rows)
    c_count = len(columns)

    # Initial safe squares
    safe_squares = (N - r_count) * (N - c_count)

    # Compute X1: diag_sum contributions
    X1 = 0
    for c in diag_sum:
        i_min = max(1, c - N)
        i_max = min(N, c - 1)
        if i_min > i_max:
            continue
        total_i = i_max - i_min + 1

        # Compute A
        left = bisect.bisect_left(rows_sorted, i_min)
        right = bisect.bisect_right(rows_sorted, i_max)
        A = right - left

        # Compute B
        B = 0
        for j in columns:
            i_candidate = c - j
            if i_min <= i_candidate <= i_max and 1 <= j <= N:
                B += 1

        # Compute AB
        AB = 0
        # Iterate through rows in [i_min, i_max]
        left_idx = bisect.bisect_left(rows_sorted, i_min)
        right_idx = bisect.bisect_right(rows_sorted, i_max)
        for i in rows_sorted[left_idx:right_idx]:
            j_candidate = c - i
            if j_candidate in columns_set and 1 <= j_candidate <= N:
                AB += 1

        count_c = total_i - (A + B - AB)
        X1 += count_c

    # Compute X2: diag_diff contributions
    X2 = 0
    for d in diag_diff:
        i_min = max(1, 1 + d)
        i_max = min(N, N + d)
        if i_min > i_max:
            continue
        total_i = i_max - i_min + 1

        # Compute A
        left = bisect.bisect_left(rows_sorted, i_min)
        right = bisect.bisect_right(rows_sorted, i_max)
        A = right - left

        # Compute B
        B = 0
        for j in columns:
            i_candidate = j + d
            if i_min <= i_candidate <= i_max and 1 <= j <= N:
                B += 1

        # Compute AB
        AB = 0
        # Iterate through rows in [i_min, i_max]
        left_idx = bisect.bisect_left(rows_sorted, i_min)
        right_idx = bisect.bisect_right(rows_sorted, i_max)
        for i in rows_sorted[left_idx:right_idx]:
            j_candidate = i - d
            if j_candidate in columns_set and 1 <= j_candidate <= N:
                AB += 1

        count_d = total_i - (A + B - AB)
        X2 += count_d

    # Compute X3: overlap between diag_sum and diag_diff
    X3 = 0
    for s in diag_sum:
        for d in diag_diff:
            if (s + d) % 2 != 0 or (s - d) % 2 != 0:
                continue
            i = (s + d) // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N:
                if i not in rows and j not in columns:
                    X3 += 1

    # Final calculation
    safe_squares -= (X1 + X2 - X3)
    print(safe_squares)

if __name__ == "__main__":
    main()