def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    steps = []
    for _ in range(N):
        A = int(input[idx])
        S = input[idx + 1]
        steps.append((A, S))
        idx += 2

    INF = float('inf')
    prev_dp = [[0] * 101 for _ in range(101)]

    for A, S in steps:
        curr_dp = [[INF] * 101 for _ in range(101)]
        if S == 'L':
            min_for_R_prev = [min(prev_dp[L_prev][r] + abs(A - L_prev) for L_prev in range(1, 101)) for r in range(1, 101)]
            for R_curr in range(1, 101):
                min_cost = min(min_for_R_prev[r_prev - 1] + abs(R_curr - r_prev) for r_prev in range(1, 101))
                curr_dp[A][R_curr] = min_cost
        else:
            min_for_L_prev = [min(prev_dp[l][R_prev] + abs(A - R_prev) for R_prev in range(1, 101)) for l in range(1, 101)]
            for L_curr in range(1, 101):
                min_cost = min(min_for_L_prev[l_prev - 1] + abs(L_curr - l_prev) for l_prev in range(1, 101))
                curr_dp[L_curr][A] = min_cost
        prev_dp = [row[:] for row in curr_dp]

    min_fatigue = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if prev_dp[l][r] < min_fatigue:
                min_fatigue = prev_dp[l][r]
    print(min_fatigue)

if __name__ == '__main__':
    main()