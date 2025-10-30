import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    b = list(map(int, data[2+n:2+2*n]))
    
    def between(x, a, b):
        if a <= b:
            return a <= x <= b
        else:
            return x >= a or x <= b
            
    def dif(x, y):
        d = abs(x - y)
        return min(d, m - d)
        
    INF = 10**18
    dp = [[INF, INF] for _ in range(n)]
    d0 = dif(a[0], b[0])
    dp[0][0] = d0
    dp[0][1] = m - d0
    
    for i in range(1, n):
        mid_val = b[i-1]
        from_val = a[i]
        to_val = b[i]
        cost0 = dif(from_val, to_val)
        cost1 = m - cost0
        
        for j in range(2):
            if dp[i-1][j] >= INF:
                continue
            for k in range(2):
                cost_add = cost0 if k == 0 else cost1
                if between(mid_val, from_val, to_val) and between(mid_val, to_val, from_val):
                    cost_add += m
                total_cost = dp[i-1][j] + cost_add
                if total_cost < dp[i][k]:
                    dp[i][k] = total_cost
                    
    ans = min(dp[n-1][0], dp[n-1][1])
    print(ans if ans < INF else -1)

if __name__ == '__main__':
    main()