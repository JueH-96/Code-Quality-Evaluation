import math

def main():
    n = int(input())
    coords = [tuple(map(int, input().split())) for _ in range(n)]
    max_c = 60
    INF = float('inf')
    
    # Initialize DP table
    dp = [ [INF] * (max_c + 1) for _ in range(n) ]
    dp[0][0] = 0.0  # Starting at checkpoint 0 with 0 skips
    
    for i in range(n):
        for c in range(max_c + 1):
            if dp[i][c] == INF:
                continue
            # Calculate the maximum j we can transition to
            max_j = i + (max_c - c) + 1
            max_j = min(max_j, n - 1)
            for j in range(i + 1, max_j + 1):
                skips = j - i - 1
                c_new = c + skips
                if c_new > max_c:
                    continue
                dx = coords[j][0] - coords[i][0]
                dy = coords[j][1] - coords[i][1]
                dist = math.hypot(dx, dy)
                if dp[j][c_new] > dp[i][c] + dist:
                    dp[j][c_new] = dp[i][c] + dist
    
    # Calculate the minimal total including penalty
    min_total = INF
    for c in range(max_c + 1):
        if dp[n-1][c] == INF:
            continue
        penalty = 0 if c == 0 else (2 ** (c - 1))
        total = dp[n-1][c] + penalty
        if total < min_total:
            min_total = total
    
    # Print with sufficient precision
    print("{0:.20f}".format(min_total))

if __name__ == "__main__":
    main()