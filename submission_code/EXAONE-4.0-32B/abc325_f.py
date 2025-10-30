import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    D_list = list(map(int, data[1:1+n]))
    L0 = int(data[1+n]); C0 = int(data[2+n]); K0 = int(data[3+n])
    L1 = int(data[4+n]); C1 = int(data[5+n]); K1 = int(data[6+n])
    
    INF = 10**18
    dp = [INF] * (K0+1)
    dp[0] = 0
    
    for d in D_list:
        if L0 == 0:
            x_i_max = 0
        else:
            x_i_max = (d + L0 - 1) // L0
        if x_i_max > K0:
            x_i_max = K0
        
        new_dp = [INF] * (K0+1)
        
        for x_i in range(0, x_i_max+1):
            residual = d - x_i * L0
            if residual <= 0:
                y_i = 0
            else:
                y_i = (residual + L1 - 1) // L1
            
            for total_x in range(x_i, K0+1):
                prev_total = total_x - x_i
                if dp[prev_total] != INF:
                    candidate = dp[prev_total] + y_i
                    if candidate < new_dp[total_x]:
                        new_dp[total_x] = candidate
        
        dp = new_dp
    
    ans = INF
    for total_x in range(0, K0+1):
        if dp[total_x] <= K1:
            cost = total_x * C0 + dp[total_x] * C1
            if cost < ans:
                ans = cost
    
    if ans == INF:
        print(-1)
    else:
        print(ans)

if __name__ == '__main__':
    main()