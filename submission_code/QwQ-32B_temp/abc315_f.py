import sys
import math

def main():
    N = int(sys.stdin.readline())
    points = [(0, 0)]  # 1-based indexing
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))
    
    max_skips = 20
    INF = float('inf')
    
    # Initialize DP table: dp[i][k] is the minimal distance to reach checkpoint i with k skips so far
    dp = [[INF] * (max_skips + 1) for _ in range(N + 1)]
    dp[1][0] = 0.0  # Starting at checkpoint 1 with 0 skips
    
    for i in range(1, N):
        for k in range(max_skips + 1):
            if dp[i][k] == INF:
                continue
            # Determine the maximum j we can consider
            max_j = i + max_skips + 1
            if max_j > N:
                max_j = N
            for j in range(i + 1, max_j + 1):
                delta_skips = j - i - 1
                new_k = k + delta_skips
                if new_k > max_skips:
                    continue
                # Calculate distance between i and j
                xi, yi = points[i]
                xj, yj = points[j]
                distance = math.hypot(xj - xi, yj - yi)
                new_dist = dp[i][k] + distance
                if new_dist < dp[j][new_k]:
                    dp[j][new_k] = new_dist
    
    # Compute the minimal s considering all possible k
    min_s = INF
    for k in range(max_skips + 1):
        if dp[N][k] == INF:
            continue
        penalty = 2 ** (k - 1) if k > 0 else 0
        total = dp[N][k] + penalty
        if total < min_s:
            min_s = total
    
    print("{0:.12f}".format(min_s))

if __name__ == "__main__":
    main()