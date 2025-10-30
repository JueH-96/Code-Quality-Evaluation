import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # last occurrence of each value, and whether we've seen it
    last = [0] * (n + 2)
    seen = [False] * (n + 2)

    distinct = 0
    sum_gaps = 0
    # Process each A_i, accumulate the "gap" contributions before each occurrence
    for i in range(1, n + 1):
        x = int(next(it))
        if not seen[x]:
            # first time we see x: initial gap = i-1
            distinct += 1
            g = i - 1
            sum_gaps += g * (g + 1) // 2
            seen[x] = True
        else:
            # gap between last[x] and i
            g = i - last[x] - 1
            sum_gaps += g * (g + 1) // 2
        last[x] = i

    # total number of subarrays
    total_subs = n * (n + 1) // 2
    # now account for the final gap after the last occurrence of each value
    for x in range(1, n + 1):
        if seen[x]:
            g = n - last[x]
            sum_gaps += g * (g + 1) // 2

    # The answer is sum_x ( #subarrays containing x )
    # = sum_x [ total_subs - #subarrays avoiding x ]
    # = distinct * total_subs - sum_over_gaps(g*(g+1)/2)
    ans = distinct * total_subs - sum_gaps
    sys.stdout.write(str(ans))


if __name__ == "__main__":
    main()