import math
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    pts = []
    idx = 1
    for i in range(n):
        x = int(data[idx])
        y = int(data[idx+1])
        idx += 2
        pts.append((x, y))
    
    T = 30
    
    if n == 1:
        print("0.00000000000000000000")
        return
        
    dp = [[10**18] * (T+1) for _ in range(n)]
    dp[0][0] = 0.0
    
    for i in range(1, n):
        low_j = max(0, i - T - 1)
        for j in range(low_j, i):
            skip_between = i - j - 1
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d_val = math.sqrt(dx*dx + dy*dy)
            for s in range(skip_between, T+1):
                s_old = s - skip_between
                candidate = dp[j][s_old] + d_val
                if candidate < dp[i][s]:
                    dp[i][s] = candidate
                    
    ans = 10**18
    max_s = min(T, n-1)
    for s in range(0, max_s+1):
        if dp[n-1][s] >= 10**18:
            continue
        penalty = 0
        if s > 0:
            penalty = 2**(s-1)
        total_cost = dp[n-1][s] + penalty
        if total_cost < ans:
            ans = total_cost
            
    print("{:.20f}".format(ans))

if __name__ == "__main__":
    main()