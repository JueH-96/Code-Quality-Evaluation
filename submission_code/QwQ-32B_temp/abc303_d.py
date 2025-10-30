def main():
    import sys
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    INF = float('inf')
    prev_dp = [0, INF]  # [state 0, state 1]

    for c in S:
        current_dp = [INF, INF]
        for s_prev in [0, 1]:
            if prev_dp[s_prev] == INF:
                continue
            for toggle in [0, 1]:
                s_new = s_prev ^ toggle
                toggle_cost = toggle * Z
                # Check action1
                if (s_new == 0 and c == 'a') or (s_new == 1 and c == 'A'):
                    cost = prev_dp[s_prev] + toggle_cost + X
                    if cost < current_dp[s_new]:
                        current_dp[s_new] = cost
                # Check action2
                if (s_new == 0 and c == 'A') or (s_new == 1 and c == 'a'):
                    cost = prev_dp[s_prev] + toggle_cost + Y
                    if cost < current_dp[s_new]:
                        current_dp[s_new] = cost
        prev_dp = current_dp.copy()

    print(min(prev_dp))

if __name__ == "__main__":
    main()