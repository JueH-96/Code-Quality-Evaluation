import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    points = []
    idx = 1
    for _ in range(n):
        x = int(data[idx])
        y = int(data[idx+1])
        points.append((x, y))
        idx += 2

    # Compute the sum of consecutive distances (S)
    S = 0.0
    for i in range(n-1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        S += math.hypot(dx, dy)

    min_s = S  # Initialize with the case where no checkpoints are skipped (C=0)

    # Consider skipping up to 20 checkpoints (C=1 to 20)
    for C in range(1, 21):
        K = n - C
        if K < 2:
            continue  # Not possible to have less than 2 checkpoints

        # Initialize DP array. dp[i] is the minimal distance to reach checkpoint i with exactly m steps
        m = K
        dp = [float('inf')] * n
        dp[0] = 0.0  # Starting at checkpoint 1 (index 0)

        for step in range(2, m + 1):
            new_dp = [float('inf')] * n
            for i in range(n):
                if dp[i] == float('inf'):
                    continue
                # Only look at checkpoints after i and within a reasonable range to optimize
                # Limit the search to the next 20 checkpoints to reduce time (heuristic)
                max_j = min(n, i + 20)
                for j in range(i + 1, max_j):
                    cost = math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
                    if new_dp[j] > dp[i] + cost:
                        new_dp[j] = dp[i] + cost
            dp = new_dp

        if dp[-1] == float('inf'):
            continue  # No valid path found for this K

        penalty = 2 ** (C - 1)
        current_s = dp[-1] + penalty
        if current_s < min_s:
            min_s = current_s

    # Ensure the output has enough precision
    print("{0:.20f}".format(min_s))

if __name__ == "__main__":
    main()