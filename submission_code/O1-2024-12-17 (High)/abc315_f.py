def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    coords = []
    idx = 1
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx+1])
        coords.append((x, y))
        idx += 2

    # If N=2, no skipping possible, answer is just the distance from 1 to 2
    if N == 2:
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        dist = math.hypot(x1 - x0, y1 - y0)
        print(f"{dist:.9f}")
        return

    # Based on reasoning, skipping beyond 34 is never beneficial due to large penalties.
    skip_max = min(N - 2, 34)

    # Precompute distances for forward jumps up to skip_max
    dist_next = [ [0.0]*(skip_max+1) for _ in range(N) ]
    for i in range(N):
        x1, y1 = coords[i]
        for d in range(1, skip_max+1):
            j = i + d
            if j >= N:
                break
            x2, y2 = coords[j]
            dist_next[i][d] = math.hypot(x2 - x1, y2 - y1)

    INF = float('inf')
    # dp[i][c] = minimal travel distance to reach checkpoint i having skipped c checkpoints in total
    dp = [ [INF]*(skip_max+1) for _ in range(N) ]
    dp[0][0] = 0.0

    # Fill DP table
    for i in range(N - 1):
        dp_i = dp[i]
        dist_i = dist_next[i]
        for c in range(skip_max + 1):
            base = dp_i[c]
            if base == INF:
                continue
            limit = skip_max - c
            # Try all possible jumps from i to i+d (1 <= d <= limit+1)
            for d in range(1, limit + 2):
                j = i + d
                if j >= N:
                    break
                nc = c + (d - 1)   # new skip count
                # Update DP if this path is better
                cost = base + dist_i[d]
                if cost < dp[j][nc]:
                    dp[j][nc] = cost

    # Penalty array
    penalty = [0.0]*(skip_max+1)
    for c in range(1, skip_max+1):
        penalty[c] = 2**(c - 1)

    # Find the minimal possible s
    ans = INF
    for c in range(skip_max + 1):
        val = dp[N - 1][c] + penalty[c]
        if val < ans:
            ans = val

    print(f"{ans:.9f}")

# Do not forget to call main()
main()