import sys

def main():
    N, K, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    
    sum_T = [0] * (N + 1)
    for i in range(1, N + 1):
        sum_T[i] = sum_T[i-1] + T[i-1]
    
    INF = 10**18
    dp = [(INF, INF)] * (N + 1)
    dp[0] = (-INF, 0)
    
    for i in range(1, N + 1):
        current_min = (INF, INF)
        start_j = max(0, i - K)
        for j in range(start_j, i):
            prev_day, prev_diss = dp[j]
            current_T_max = T[i-1]
            required_S = max(current_T_max, prev_day + X)
            num_orders = i - j
            sum_part = sum_T[i] - sum_T[j]
            added_diss = num_orders * required_S - sum_part
            total_diss = prev_diss + added_diss
            candidate = (required_S, total_diss)
            if (candidate[1] < current_min[1]) or \
               (candidate[1] == current_min[1] and candidate[0] < current_min[0]):
                current_min = candidate
        dp[i] = current_min
    
    print(int(dp[N][1]))

if __name__ == "__main__":
    main()