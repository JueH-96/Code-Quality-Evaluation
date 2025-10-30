import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    k = int(data[1])
    x = int(data[2])
    T = list(map(int, data[3:3+n]))
    
    prefix = [0] * (n+1)
    for i in range(1, n+1):
        prefix[i] = prefix[i-1] + T[i-1]
        
    seg_max = [[0] * n for _ in range(n)]
    for l in range(n):
        cur_max = T[l]
        for r in range(l, n):
            if T[r] > cur_max:
                cur_max = T[r]
            seg_max[l][r] = cur_max
            
    INF = 10**18
    dp = [[(INF, 0)] * (n+1) for _ in range(n+1)]
    dp[0][0] = (0, -x)
    
    for i in range(1, n+1):
        for g in range(1, i+1):
            low_j = max(0, i - k)
            for j in range(low_j, i):
                if dp[j][g-1][0] == INF:
                    continue
                M = seg_max[j][i-1]
                prev_last = dp[j][g-1][1]
                d_current = max(M, prev_last + x)
                cnt = i - j
                group_sum = prefix[i] - prefix[j]
                cost_group = cnt * d_current - group_sum
                total_dissatisfaction = dp[j][g-1][0] + cost_group
                if total_dissatisfaction < dp[i][g][0]:
                    dp[i][g] = (total_dissatisfaction, d_current)
                    
    ans = min(dp[n][g][0] for g in range(1, n+1))
    print(ans)

if __name__ == "__main__":
    main()