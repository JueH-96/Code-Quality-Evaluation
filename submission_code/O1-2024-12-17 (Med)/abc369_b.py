def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = []
    S = []
    idx = 1
    for _ in range(N):
        A.append(int(input_data[idx]))
        S.append(input_data[idx+1])
        idx += 2

    # We will use a rolling DP approach to reduce memory usage.
    # dp[l][r] will represent the minimum fatigue so far with
    # left hand on key l, right hand on key r.
    INF = float('inf')
    prev_dp = [[0]*101 for _ in range(101)]  # dp after 0 presses: cost = 0 for any hand placement
    # Actually, to start "anywhere" at no cost, we must set them all to 0
    # because we can initially place hands on any keys with no cost.
    # This is already done as we have initialized everything to 0.

    for i in range(1,101):
        for j in range(1,101):
            prev_dp[i][j] = 0

    # Process each press
    for i in range(N):
        cur_dp = [[INF]*101 for _ in range(101)]
        key = A[i]
        hand = S[i]
        for l in range(1,101):
            for r in range(1,101):
                cost_so_far = prev_dp[l][r]
                if cost_so_far == INF:
                    continue
                if hand == 'L':
                    # Move left hand from l to key
                    move_cost = abs(key - l)
                    if cost_so_far + move_cost < cur_dp[key][r]:
                        cur_dp[key][r] = cost_so_far + move_cost
                else:  # hand == 'R'
                    # Move right hand from r to key
                    move_cost = abs(key - r)
                    if cost_so_far + move_cost < cur_dp[l][key]:
                        cur_dp[l][key] = cost_so_far + move_cost
        prev_dp = cur_dp

    # The answer is the minimum fatigue among all possible end positions
    ans = INF
    for l in range(1,101):
        for r in range(1,101):
            ans = min(ans, prev_dp[l][r])

    print(ans)

# Do not forget to call main function
if __name__ == "__main__":
    main()