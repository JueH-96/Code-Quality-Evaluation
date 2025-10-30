# YOUR CODE HERE
# Read the inputs
import sys
import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)

    N, K, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    T.sort()
    N = len(T)
    max_T = T[-1]

    # Collect possible days
    D_set = set(T)
    days_to_add = set(T)
    max_limit = T[-1] + N * X + X

    while days_to_add:
        new_days = set()
        for day in days_to_add:
            next_day = day + X
            if next_day <= max_limit:
                if next_day not in D_set:
                    new_days.add(next_day)
                    D_set.add(next_day)
        days_to_add = new_days

    D = sorted(D_set)
    day_to_idx = {day: idx for idx, day in enumerate(D)}
    M = len(D)

    # Precompute mapping from s_idx to s_idx_after (day + X)
    s_idx_after = [-1] * M
    for idx in range(M):
        day = D[idx]
        next_day = day + X
        next_idx = bisect.bisect_left(D, next_day)
        if next_idx < M and D[next_idx] == next_day:
            s_idx_after[idx] = next_idx

    # Precompute prefix sums of T_i
    prefix_sum_T = [0] * (N + 1)
    for i in range(N):
        prefix_sum_T[i + 1] = prefix_sum_T[i] + T[i]

    # Initialize DP table
    INF = float('inf')
    dp = [[INF] * M for _ in range(N + 1)]  # dp[i][d_idx]
    # Base case
    for d_idx in range(M):
        dp[N][d_idx] = 0

    # Build list of possible shipping days for orders
    possible_s_idx_for_order = [bisect.bisect_left(D, T_i) for T_i in T]

    # DP computation
    for i in range(N - 1, -1, -1):
        for d_idx in range(M):
            curr_dp = dp[i + 1][d_idx]
            if curr_dp == INF:
                continue
            for sz in range(1, K + 1):
                j = i + sz - 1
                if j >= N:
                    break
                earliest_possible_day = max(D[d_idx], T[j])
                s_idx = bisect.bisect_left(D, earliest_possible_day)
                if s_idx == M:
                    continue
                day_s = D[s_idx]
                dis = day_s * sz - (prefix_sum_T[j + 1] - prefix_sum_T[i])
                s_idx_next = s_idx_after[s_idx]
                if s_idx_next == -1:
                    continue
                if dp[i][s_idx_next] > curr_dp + dis:
                    dp[i][s_idx_next] = curr_dp + dis

    # Obtain the minimal total dissatisfaction
    answer = INF
    for d_idx in range(M):
        if dp[0][d_idx] < answer:
            answer = dp[0][d_idx]
    print(int(answer))

threading.Thread(target=main).start()