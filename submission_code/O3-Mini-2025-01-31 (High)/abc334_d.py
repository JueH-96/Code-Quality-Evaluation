def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    if not data:
        return

    # Read N and Q
    N = int(data[0])
    Q = int(data[1])

    # Read the reindeer requirements for each sleigh and sort them
    R = list(map(int, data[2:2+N]))
    R.sort()

    # Compute prefix sums so that prefix[i] is the sum of the smallest i+1 requirements.
    prefix = [0] * N
    prefix[0] = R[0]
    for i in range(1, N):
        prefix[i] = prefix[i - 1] + R[i]

    # Process queries
    out = []
    offset = 2 + N
    for q in range(Q):
        X = int(data[offset + q])
        # Use binary search to determine the maximum number of sleighs
        # that can be pulled with X reindeer. We search for the rightmost position
        # we can insert X in the prefix sum array without breaking the order.
        ans = bisect.bisect_right(prefix, X)
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()