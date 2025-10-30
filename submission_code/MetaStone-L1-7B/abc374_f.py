def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    X = int(input[2])
    T = list(map(int, input[3:3+N]))
    
    prefix_sum = [0] * (N + 1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + T[i-1]
    
    dp = [float('inf')] * (N + 1)
    dp[0] = 0
    
    for i in range(1, N + 1):
        for j in range(max(0, i - K), i):
            group_size = i - j
            if group_size > K:
                continue
            a = j + 1
            b = i
            cost_j = (b - a + 1) * T[b-1] - (prefix_sum[b] - prefix_sum[a-1])
            next_day = T[b-1] + X
            next_i = i + 1
            if next_i > N:
                next_min = float('inf')
            else:
                next_min = float('inf')
                for l in range(next_i, min(next_i + K, N + 1)):
                    if l > N:
                        break
                    if T[l-1] >= next_day:
                        cost2 = (l - next_i + 1) * T[l-1] - (prefix_sum[l] - prefix_sum[next_i-1])
                        total = dp[j] + cost_j + cost2
                        if total < next_min:
                            next_min = total
            if next_min != float('inf') and dp[i] > dp[j] + cost_j + next_min:
                dp[i] = dp[j] + cost_j + next_min
    
    print(dp[N])

if __name__ == '__main__':
    main()