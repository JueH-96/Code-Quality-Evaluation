# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1000000)
    N, K, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + T[i]
    
    # Precompute max_T[i][j] for i <= j < N
    max_T = [[0]*N for _ in range(N)]
    for i in range(N):
        current_max = T[i]
        for j in range(i, N):
            if T[j] > current_max:
                current_max = T[j]
            max_T[i][j] = current_max

    INF = float('inf')
    dp = [dict() for _ in range(N+1)]
    dp[0][0] = 0

    for i in range(N):
        for last, cost in dp[i].items():
            for group_size in range(1, K+1):
                j = i + group_size
                if j > N:
                    break
                max_T_group = max_T[i][j-1]
                sum_T_group = prefix_sum[j] - prefix_sum[i]
                S = max(max_T_group, last + X)
                dissatisfaction = group_size * S - sum_T_group
                new_cost = cost + dissatisfaction
                if S in dp[j]:
                    if new_cost < dp[j][S]:
                        dp[j][S] = new_cost
                else:
                    dp[j][S] = new_cost

    min_total_dissatisfaction = min(dp[N].values())
    print(min_total_dissatisfaction)

if __name__ == "__main__":
    main()