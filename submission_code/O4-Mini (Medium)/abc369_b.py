def main():
    import sys
    input = sys.stdin.readline
    INF = 10**18

    N = int(input().strip())
    presses = []
    for _ in range(N):
        a, s = input().split()
        presses.append((int(a), s))

    # dp[l][r] = minimum fatigue after processing up to current press,
    # with left hand at l and right hand at r.
    # We only keep current dp and next dp.
    # Keys are 1..100
    dp = [[INF] * 101 for _ in range(101)]

    # Initialize for the first press
    a1, s1 = presses[0]
    if s1 == 'L':
        # left goes to a1, right can be anywhere (1..100)
        for r in range(1, 101):
            dp[a1][r] = 0
    else:
        # right goes to a1, left can be anywhere
        for l in range(1, 101):
            dp[l][a1] = 0

    # Process subsequent presses
    for i in range(1, N):
        ai, si = presses[i]
        # next dp
        nxt = [[INF] * 101 for _ in range(101)]
        if si == 'L':
            # move left hand to ai
            for l in range(1, 101):
                # previous left was at l, new left will be at ai
                # cost of moving = |ai - l|
                cost_move = abs(ai - l)
                dp_l = dp[l]
                # r loops
                for r in range(1, 101):
                    prev_cost = dp_l[r]
                    if prev_cost == INF:
                        continue
                    new_cost = prev_cost + cost_move
                    if new_cost < nxt[ai][r]:
                        nxt[ai][r] = new_cost
        else:
            # move right hand to ai
            for r in range(1, 101):
                cost_move = abs(ai - r)
                # for each left position
                for l in range(1, 101):
                    prev_cost = dp[l][r]
                    if prev_cost == INF:
                        continue
                    new_cost = prev_cost + cost_move
                    if new_cost < nxt[l][ai]:
                        nxt[l][ai] = new_cost
        dp = nxt

    # Answer is the minimum over all (l, r)
    ans = INF
    for l in range(1, 101):
        row = dp[l]
        for r in range(1, 101):
            if row[r] < ans:
                ans = row[r]

    print(ans)

if __name__ == "__main__":
    main()