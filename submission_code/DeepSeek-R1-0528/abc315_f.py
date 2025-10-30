import math
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index+1])
        index += 2
        points.append((x, y))
        
    def dist(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    INF = 10**18
    dp = [[INF] * 21 for _ in range(n)]
    dp[0][0] = 0.0
    
    for i in range(1, n):
        max_skip_here = min(20, i - 1)
        for k_skip in range(0, max_skip_here + 1):
            start_j = max(0, i - 21)
            for j in range(start_j, i):
                g = i - j - 1
                if g > 20:
                    continue
                k_prev = k_skip - g
                if k_prev < 0:
                    continue
                if dp[j][k_prev] < INF:
                    new_val = dp[j][k_prev] + dist(j, i)
                    if new_val < dp[i][k_skip]:
                        dp[i][k_skip] = new_val
                        
    ans = INF
    max_total_skip = min(20, n - 2)
    for k_skip in range(0, max_total_skip + 1):
        penalty = 0
        if k_skip > 0:
            penalty = 2 ** (k_skip - 1)
        total_val = dp[n-1][k_skip] + penalty
        if total_val < ans:
            ans = total_val
            
    print("{:.20f}".format(ans))

if __name__ == "__main__":
    main()