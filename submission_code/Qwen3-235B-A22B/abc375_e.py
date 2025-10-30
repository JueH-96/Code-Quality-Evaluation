def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    people = []
    sum_b = 0
    for _ in range(N):
        a = int(input[idx])
        b = int(input[idx+1])
        people.append((a, b))
        sum_b += b
        idx += 2
    
    if sum_b % 3 != 0:
        print(-1)
        return
    
    T = sum_b // 3
    INF = float('inf')
    
    # Initialize DP table
    dp = [[INF] * (T + 1) for _ in range(T + 1)]
    dp[0][0] = 0
    
    for (a, b) in people:
        new_dp = [[INF] * (T + 1) for _ in range(T + 1)]
        for s1 in range(T + 1):
            for s2 in range(T + 1):
                if dp[s1][s2] == INF:
                    continue
                # Assign to team 1
                new_s1 = s1 + b
                new_s2 = s2
                if new_s1 <= T:
                    cost = dp[s1][s2] + (0 if a == 1 else 1)
                    if cost < new_dp[new_s1][new_s2]:
                        new_dp[new_s1][new_s2] = cost
                # Assign to team 2
                new_s1 = s1
                new_s2 = s2 + b
                if new_s2 <= T:
                    cost = dp[s1][s2] + (0 if a == 2 else 1)
                    if cost < new_dp[new_s1][new_s2]:
                        new_dp[new_s1][new_s2] = cost
                # Assign to team 3
                cost = dp[s1][s2] + (0 if a == 3 else 1)
                if cost < new_dp[s1][s2]:
                    new_dp[s1][s2] = cost
        dp = new_dp
    
    if dp[T][T] < INF:
        print(dp[T][T])
    else:
        print(-1)

if __name__ == "__main__":
    main()