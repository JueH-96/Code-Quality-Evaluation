import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    D = list(map(int, data[1:1+n]))
    L1, C1, K1 = map(int, data[1+n:1+n+3])
    L2, C2, K2 = map(int, data[1+n+3:1+n+6])
    
    INF = 10**18
    dp = [INF] * (K1+1)
    dp[0] = 0
    
    for d in D:
        k0 = (d + L1 - 1) // L1
        max_k = min(K1, k0)
        
        b_arr = []
        for k in range(0, max_k+1):
            if k * L1 >= d:
                b_arr.append(0)
            else:
                remaining = d - k * L1
                b_val = (remaining + L2 - 1) // L2
                b_arr.append(b_val)
        
        new_dp = [INF] * (K1+1)
        
        for k in range(0, max_k+1):
            b_val = b_arr[k]
            for a in range(k, K1+1):
                if dp[a - k] == INF:
                    continue
                candidate = dp[a - k] + b_val
                if candidate < new_dp[a]:
                    new_dp[a] = candidate
                    
        dp = new_dp
    
    ans = INF
    for a in range(0, K1+1):
        if dp[a] <= K2:
            total_cost = a * C1 + dp[a] * C2
            if total_cost < ans:
                ans = total_cost
                
    print(-1 if ans == INF else ans)

if __name__ == "__main__":
    main()