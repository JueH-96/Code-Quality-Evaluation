def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    points = []
    for _ in range(n):
        x = float(next(it))
        y = float(next(it))
        points.append((x, y))
    
    # dp[i] = minimum achievable value (distance plus penalties) to reach checkpoint i (0-indexed)
    INF = float('inf')
    dp = [INF] * n
    dp[0] = 0.0

    # Note: When we jump from checkpoint i to j, we skip (j-i-1) checkpoints.
    #   If we skip C > 0 checkpoints, the penalty is 2^(C-1) and 0 if C == 0.
    # Thus, if gap = j-i, then:
    #   if gap == 1: penalty = 0
    #   else: penalty = 2^( (j-i-1) - 1 ) = 2^(j-i-2)
    #
    # Since the penalty grows exponentially and the maximum Euclidean distance is limited by 
    # the point coordinates, it is very unlikely to benefit from very long jumps.
    # We thus only need to check a limited number of previous checkpoints.
    #
    # Because coordinates are <= 10^4, the maximum Euclidean distance is ~14142.
    # For a jump with gap k, the penalty is 2^(k-2) (for k>=2).
    # Even for k=17, the penalty is 2^(15)=32768, which is far larger than any possible saved distance.
    #
    # Therefore, we limit our search to a safe window. Here we choose max_gap = 20.
    
    max_gap = 20

    for j in range(1, n):
        # Only check previous checkpoints within the safe window.
        start = max(0, j - max_gap)
        best = INF
        xj, yj = points[j]
        for i in range(start, j):
            xi, yi = points[i]
            # Euclidean distance from checkpoint i to checkpoint j
            dist = math.hypot(xj - xi, yj - yi)
            gap = j - i
            if gap == 1:
                pen = 0
            else:
                pen = 2 ** (gap - 2)
            cost = dp[i] + dist + pen
            if cost < best:
                best = cost
        dp[j] = best

    # We are required to output with a tolerance of 1e-5.
    sys.stdout.write("{:.10f}".format(dp[n - 1]))
    
if __name__ == '__main__':
    main()