def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = []
    B = []
    idx = 1
    
    for _ in range(N):
        a = int(data[idx])
        b = int(data[idx + 1])
        idx += 2
        A.append(a)
        B.append(b)

    total_strength = sum(B)
    
    # If total strength is not divisible by 3, it's impossible
    if total_strength % 3 != 0:
        print(-1)
        return

    S = total_strength // 3

    # Prepare prefix sums to quickly get the sum of the first i people
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + B[i]

    INF = 10**9
    # dp[x][y] will hold the minimum number of switches needed
    # to assign the first i people so that:
    #   - team1's strength = x
    #   - team2's strength = y
    #   - team3's strength = prefix_sum[i] - x - y
    # Initially no people assigned => sum1 = 0, sum2 = 0 => cost = 0
    dp = [[INF] * (S + 1) for _ in range(S + 1)]
    dp[0][0] = 0

    for i in range(N):
        next_dp = [[INF] * (S + 1) for _ in range(S + 1)]
        b = B[i]          # strength of person i
        a = A[i]          # original team of person i
        p_sum = prefix_sum[i + 1]  # sum of strengths up to person i (inclusive)

        for sum1 in range(S + 1):
            row_dp = dp[sum1]
            for sum2 in range(S + 1):
                cost = row_dp[sum2]
                if cost == INF:
                    continue

                # Option 1: Assign current person i to team 1
                new_sum1 = sum1 + b
                if new_sum1 <= S:
                    new_sum3 = p_sum - new_sum1 - sum2
                    if new_sum3 <= S:
                        cost_move = 0 if a == 1 else 1
                        new_val = cost + cost_move
                        if new_val < next_dp[new_sum1][sum2]:
                            next_dp[new_sum1][sum2] = new_val

                # Option 2: Assign current person i to team 2
                new_sum2 = sum2 + b
                if new_sum2 <= S:
                    new_sum3 = p_sum - sum1 - new_sum2
                    if new_sum3 <= S:
                        cost_move = 0 if a == 2 else 1
                        new_val = cost + cost_move
                        if new_val < next_dp[sum1][new_sum2]:
                            next_dp[sum1][new_sum2] = new_val

                # Option 3: Assign current person i to team 3
                new_sum3 = p_sum - sum1 - sum2
                if new_sum3 <= S:
                    cost_move = 0 if a == 3 else 1
                    new_val = cost + cost_move
                    old_val = next_dp[sum1][sum2]
                    if new_val < old_val:
                        next_dp[sum1][sum2] = new_val

        dp = next_dp

    ans = dp[S][S]
    print(ans if ans != INF else -1)

# Don't forget to call main()
if __name__ == "__main__":
    main()