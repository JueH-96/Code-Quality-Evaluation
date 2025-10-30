def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    X = int(input_data[2])
    T = list(map(int, input_data[3:]))

    # Prefix sums for T to quickly compute sum of T[a..b].
    # P[i] = T[0] + T[1] + ... + T[i-1], so sum of T[j..i-1] = P[i] - P[j].
    P = [0] * (N+1)
    for i in range(N):
        P[i+1] = P[i] + T[i]
    
    # cost[i][c] = minimal total dissatisfaction when shipping first i orders in exactly c batches.
    # dp[i][c]   = the day on which we ship the c-th batch in that minimal-dissatisfaction schedule.
    # We will then take the minimum of cost[N][c] over c=1..N as our final answer.
    
    INF = 10**24
    cost = [[INF]*(N+1) for _ in range(N+1)]
    day_ship = [[INF]*(N+1) for _ in range(N+1)]  # dp array to store last shipping day
    
    # Base cases:
    # For c=0, shipping 0 orders has 0 dissatisfaction and no "last shipping day".
    cost[0][0] = 0
    day_ship[0][0] = 0  # not used but keep as 0
    
    # For c=1, we consider shipping first i orders all at once in a single batch.
    # The day to ship them must be at least T[i-1]. We choose day = T[i-1].
    # The dissatisfaction = sum(day - T_j) for j=1..i => i*day - sum(T_j).
    # But we can do it in general logic inside the nested loops below.

    for c in range(1, N+1):
        # If c=1, we can fill cost[i][1] directly, or let the general logic handle it.
        for i in range(1, N+1):
            if c == 1:
                # One batch for all i orders => day = T[i-1]
                # dissatisfaction = i * T[i-1] - (T[0] + ... + T[i-1])
                # cost = that dissatisfaction
                d = T[i-1]  # shipping day
                dissatisfaction = i * d - (P[i] - P[0])
                cost[i][c] = dissatisfaction
                day_ship[i][c] = d
            else:
                # We try to form the c-th batch from some j+1..i (m = i-j orders, up to K),
                # Then day = max(day_ship[j][c-1] + X, T[i-1]).
                # dissatisfaction = (number_of_orders_in_batch)*day - sum_of_those_orders' T-values
                # total = cost[j][c-1] + dissatisfaction
                # we pick the min.
                for m in range(1, K+1):
                    j = i - m
                    if j < 0:
                        break
                    if cost[j][c-1] == INF:
                        continue
                    prev_day = day_ship[j][c-1]
                    
                    # shipping day for batch c
                    # must be at least T[i-1] (since T is 0-based, T[i-1] is T_i in 1-based logic)
                    # must also be at least prev_day + X
                    proposed_day = max(prev_day + X, T[i-1])
                    batch_diss = m * proposed_day - (P[i] - P[j])
                    total_diss = cost[j][c-1] + batch_diss
                    if total_diss < cost[i][c]:
                        cost[i][c] = total_diss
                        day_ship[i][c] = proposed_day
    
    ans = min(cost[N][1:])  # minimal dissatisfaction among 1..N batches
    print(ans)