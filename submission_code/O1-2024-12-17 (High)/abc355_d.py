def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    intervals = []
    idx = 1

    # Read intervals
    for _ in range(N):
        l = int(data[idx]); r = int(data[idx+1])
        idx += 2
        intervals.append((l, r))

    # Sort intervals by their left endpoint
    intervals.sort(key=lambda x: x[0])

    # Separate left endpoints and right endpoints into arrays
    L = [it[0] for it in intervals]
    R = [it[1] for it in intervals]

    ans = 0
    # For each interval in sorted-by-L order,
    # count how many intervals start no later than this one's right endpoint.
    for i in range(N):
        # j is the largest index s.t. L[j] <= R[i]
        # bisect_right gives the index where R[i] could be inserted to keep L sorted,
        # so subtract 1 to get the largest valid index
        j = bisect.bisect_right(L, R[i]) - 1
        # We want pairs (i, j) with j > i, so count j - i if j > i
        if j > i:
            ans += j - i

    print(ans)

# Do not forget to call main()!
main()