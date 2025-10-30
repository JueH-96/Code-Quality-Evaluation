def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    R = list(map(int, data[2:2+N]))
    queries = list(map(int, data[2+N:]))

    # Sort sleigh requirements
    R.sort()

    # Precompute prefix sums of the sorted requirements
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + R[i]

    # For each query, determine the maximum number of sleds
    # that fit into X reindeers using binary search
    results = []
    for X in queries:
        # bisect_right(prefix_sums, X) gives the insertion position
        # of X in prefix_sums to keep it sorted, so we subtract 1
        # to get how many sleighs can be pulled.
        idx = bisect.bisect_right(prefix_sums, X) - 1
        results.append(str(idx if idx >= 0 else 0))

    print("
".join(results))

# Do not forget to call main()!
main()