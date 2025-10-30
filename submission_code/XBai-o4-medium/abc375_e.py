def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    people = []
    for _ in range(N):
        a = int(input[idx])
        b = int(input[idx+1])
        people.append((a, b))
        idx += 2
    
    total_sum = sum(b for a, b in people)
    if total_sum % 3 != 0:
        print(-1)
        return
    T = total_sum // 3
    INF = 10**18
    
    # Initialize DP table
    current_dp = [[INF] * (T + 1) for _ in range(T + 1)]
    current_dp[0][0] = 0
    
    for i in range(N):
        a_i, b_i = people[i]
        next_dp = [[INF] * (T + 1) for _ in range(T + 1)]
        for s1 in range(T + 1):
            for s2 in range(T + 1):
                if current_dp[s1][s2] == INF:
                    continue
                # Assign to team 1
                new_s1 = s1 + b_i
                new_s2_ = s2
                if new_s1 <= T and new_s2_ <= T:
                    if a_i == 1:
                        changes = current_dp[s1][s2]
                    else:
                        changes = current_dp[s1][s2] + 1
                    if next_dp[new_s1][new_s2_] > changes:
                        next_dp[new_s1][new_s2_] = changes
                # Assign to team 2
                new_s1_ = s1
                new_s2 = s2 + b_i
                if new_s1_ <= T and new_s2 <= T:
                    if a_i == 2:
                        changes = current_dp[s1][s2]
                    else:
                        changes = current_dp[s1][s2] + 1
                    if next_dp[new_s1_][new_s2] > changes:
                        next_dp[new_s1_][new_s2] = changes
                # Assign to team 3
                new_s1_ = s1
                new_s2_ = s2
                if new_s1_ <= T and new_s2_ <= T:
                    if a_i == 3:
                        changes = current_dp[s1][s2]
                    else:
                        changes = current_dp[s1][s2] + 1
                    if next_dp[new_s1_][new_s2_] > changes:
                        next_dp[new_s1_][new_s2_] = changes
        current_dp = next_dp
    
    result = current_dp[T][T]
    print(-1 if result == INF else result)

if __name__ == "__main__":
    main()