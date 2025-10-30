def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    from collections import defaultdict
    # Dictionary to store indices (1-based) of each distinct value
    positions = defaultdict(list)
    for i, val in enumerate(A):
        positions[val].append(i + 1)

    answer = 0
    for val, idx_list in positions.items():
        m = len(idx_list)
        # If a value appears fewer than 2 times, it cannot form a valid (i, k) pair
        if m < 2:
            continue

        # Compute helper sums:
        # M'_m = sum_{r=0..m-1} (idx_list[r] * r)
        # N'_m = sum_{r=0..m-1} (idx_list[r] * (m-1 - r))
        M = 0
        N = 0
        for r in range(m):
            M += idx_list[r] * r
            N += idx_list[r] * (m - 1 - r)

        # #pairs = m*(m-1)/2
        pairs = m * (m - 1) // 2

        # G = sum_{1 <= p < q <= m} (q - p - 1) = m*(m-1)*(m-2)/6 for m >= 2
        G = 0
        if m >= 3:
            G = m * (m - 1) * (m - 2) // 6

        # T = sum_{p<q} (i_q - i_p - 1) = (sum_{q<r} i_q) - (sum_{p<q} i_p) - #pairs
        # using prefix sums replaced by M and N
        T = M - N - pairs

        # The number of valid triples for this particular value = T - G
        answer += (T - G)

    print(answer)

# Do not forget to call main().
if __name__ == "__main__":
    main()