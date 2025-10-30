def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    items = []
    total = 0
    for _ in range(N):
        A = int(next(it)); B = int(next(it))
        items.append((A, B))
        total += B

    # If total strength not divisible by 3, impossible
    if total % 3 != 0:
        print(-1)
        return
    t = total // 3

    # Sort by strength descending to reach sum t quickly in DP
    items.sort(key=lambda x: x[1], reverse=True)

    INF = N + 1
    # dp[a][b] = minimum moves so that team1 sum = a, team2 sum = b
    dp = [[INF] * (t + 1) for _ in range(t + 1)]
    dp[0][0] = 0

    sum_old = 0
    for A, B in items:
        w = B
        # cost of assigning this person to team1, team2, team3
        if A == 1:
            c1, c2, c3 = 0, 1, 1
        elif A == 2:
            c1, c2, c3 = 1, 0, 1
        else:
            c1, c2, c3 = 1, 1, 0

        # we only need to iterate a up to min(sum_old, t)
        so = sum_old if sum_old < t else t
        for a in range(so, -1, -1):
            row_dp = dp[a]
            rem = sum_old - a
            if rem > t:
                rem = t
            for b in range(rem, -1, -1):
                old = row_dp[b]
                if old == INF:
                    continue
                # assign to team1
                na = a + w
                if na <= t:
                    dpa = dp[na]
                    val = old + c1
                    if val < dpa[b]:
                        dpa[b] = val
                # assign to team2
                nb = b + w
                if nb <= t:
                    val2 = old + c2
                    if val2 < row_dp[nb]:
                        row_dp[nb] = val2
                # assign to team3
                row_dp[b] = old + c3

        sum_old += w

    ans = dp[t][t]
    if ans > N:
        print(-1)
    else:
        print(ans)

# call main
main()