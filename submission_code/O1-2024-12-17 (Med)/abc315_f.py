def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = []
    idx = 1
    for _ in range(N):
        x = int(input_data[idx]); y = int(input_data[idx+1])
        coords.append((x, y))
        idx += 2

    # If N=2, the answer is simply the distance between checkpoint 1 and 2 (no skip penalty).
    if N == 2:
        dx = coords[0][0] - coords[1][0]
        dy = coords[0][1] - coords[1][1]
        ans = math.hypot(dx, dy)
        print(f"{ans:.9f}")
        return

    # We will only allow skipping up to K intermediate points (K <= 25), because 2^(K-1) grows very fast
    # and skipping more generally becomes suboptimal. Also, K <= (N-2) anyway (can't skip more than all intermediates).
    max_skip = min(25, N - 2)

    # Precompute distance from i to i+k for k up to 25 (where valid).
    # dist[i][k] = distance between checkpoint i and i+k
    dist = [[0.0]*26 for _ in range(N)]
    for i in range(N):
        x1, y1 = coords[i]
        for k in range(1, 26):
            j = i + k
            if j >= N:
                break
            x2, y2 = coords[j]
            dx = x2 - x1
            dy = y2 - y1
            dist[i][k] = math.hypot(dx, dy)

    # dp[i][c] = minimum total (Euclidean) distance traveled to reach checkpoint i
    #            having skipped c checkpoints so far
    # We will fill this table in a forward manner.
    INF = float('inf')
    dp = [[INF]*(max_skip+1) for _ in range(N)]
    dp[0][0] = 0.0  # start at checkpoint 0 (which is checkpoint 1 in the problem), no distance yet

    # Fill DP
    for i in range(N-1):
        for c in range(max_skip+1):
            current_cost = dp[i][c]
            if current_cost == INF:
                continue
            # We can jump ahead k checkpoints (i -> i+k),
            # which means skipping k-1 intermediate points.
            # So our new skip count is c_new = c + (k-1).
            # We only consider up to 25 steps ahead (and within array bounds)
            # and we cannot exceed max_skip skipping.
            max_k = min(max_skip - c + 1, N - 1 - i, 25)
            for k in range(1, max_k+1):
                c_new = c + (k - 1)
                new_cost = current_cost + dist[i][k]
                if new_cost < dp[i+k][c_new]:
                    dp[i+k][c_new] = new_cost

    # Compute the final answer by adding the skip penalty.
    # Penalty = 0 if c=0, else 2^(c-1).
    best = INF
    for c in range(max_skip+1):
        penalty = 0.0 if c == 0 else 2.0**(c-1)
        val = dp[N-1][c] + penalty
        if val < best:
            best = val

    print(f"{best:.9f}")

# Do not forget to call main()
if __name__ == "__main__":
    main()