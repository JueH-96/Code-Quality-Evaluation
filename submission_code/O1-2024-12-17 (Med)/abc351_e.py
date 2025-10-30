def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])

    # We will group points by parity of (x + y).
    # Within each parity group, distance between any two points (x1, y1) and (x2, y2)
    # is ((|S1 - S2|) + (|D1 - D2|)) // 2 where
    # S = x + y, D = x - y.
    # Across different parity groups, distance is 0.

    group0_S = []
    group0_D = []
    group1_S = []
    group1_D = []

    idx = 1
    for _ in range(N):
        x = int(input_data[idx]); y = int(input_data[idx+1])
        idx += 2
        if ((x + y) & 1) == 0:
            group0_S.append(x + y)
            group0_D.append(x - y)
        else:
            group1_S.append(x + y)
            group1_D.append(x - y)

    # Helper to compute sum of pairwise |A_i - A_j| for i < j, array A.
    # This is standard: after sorting, sum_{i<j} (A_j - A_i).
    # We'll do it in O(k) using prefix sums.
    def sum_of_abs_diff(arr):
        arr.sort()
        prefix = 0
        total = 0
        for i, val in enumerate(arr):
            total += val * i - prefix
            prefix += val
        return total

    # Sum of all pairwise (|S_i - S_j| + |D_i - D_j|) within one parity group
    def sum_in_group(S, D):
        return sum_of_abs_diff(S) + sum_of_abs_diff(D)

    sum_group0 = sum_in_group(group0_S, group0_D)
    sum_group1 = sum_in_group(group1_S, group1_D)

    # Each pair's distance is ((|S_i - S_j| + |D_i - D_j|)) / 2
    # so we divide the totals by 2 at the end:
    answer = (sum_group0 + sum_group1) // 2

    print(answer)

# Do not forget to call main()!
if __name__ == "__main__":
    main()