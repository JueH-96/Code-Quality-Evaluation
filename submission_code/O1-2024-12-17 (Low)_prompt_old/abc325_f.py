def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    D = list(map(int, input_data[1:N+1]))
    L1, C1, K1 = map(int, input_data[N+1:N+4])
    L2, C2, K2 = map(int, input_data[N+4:N+7])

    # We use a DP approach where:
    #   dp[u1] = the minimum total number of type-2 sensors needed
    #            to cover the sections processed so far, if we have
    #            used exactly u1 type-1 sensors so far.
    #
    # Initially dp[0] = 0 (no sections covered, no sensors used).
    # If dp[u1] = X, then when covering the next section of length d,
    # if we decide to use x more type-1 sensors, then we must use
    # y = min number of type-2 sensors to cover any leftover length.
    # i.e. x*L1 + y*L2 >= d => y = max(0, ceil( (d - x*L1) / L2 )).
    # So newdp[u1 + x] = min( newdp[u1 + x], dp[u1] + y ).
    #
    # After processing all N sections, we look for any 0 <= u1 <= K1
    # such that dp[u1] <= K2. Among those, the cost is u1*C1 + dp[u1]*C2.
    # We take the minimum such cost or output -1 if none are feasible.

    INF = float('inf')
    dp = [INF] * (K1 + 1)
    dp[0] = 0  # no sections covered yet

    for d in D:
        newdp = [INF] * (K1 + 1)
        for used1 in range(K1 + 1):
            used2_so_far = dp[used1]
            if used2_so_far == INF:
                continue

            # Maximum additional type-1 sensors we can still use
            leftover1 = K1 - used1
            # We only need enough type-1 sensors up to covering d alone if we wanted
            # x_max covers the case x*L1 >= d, which is x >= (d + L1 - 1)//L1
            x_max = (d + L1 - 1) // L1
            if x_max > leftover1:
                x_max = leftover1

            base2 = used2_so_far
            # Try all x in [0..x_max]
            # y = max(0, ceil((d - x*L1)/L2))
            for x in range(x_max + 1):
                cover_len = x * L1
                if cover_len >= d:
                    y = 0
                else:
                    need = d - cover_len
                    y = (need + L2 - 1) // L2

                new_used1 = used1 + x
                new_used2 = base2 + y
                if new_used2 < newdp[new_used1]:
                    newdp[new_used1] = new_used2

        dp = newdp

    # Find the minimal cost among all dp[u1] that is <= K2
    ans = None
    for u1 in range(K1 + 1):
        u2 = dp[u1]
        if u2 <= K2:
            cost = u1 * C1 + u2 * C2
            if ans is None or cost < ans:
                ans = cost

    if ans is None:
        print(-1)
    else:
        print(ans)

# Call solve() after definition.
if __name__ == "__main__":
    solve()