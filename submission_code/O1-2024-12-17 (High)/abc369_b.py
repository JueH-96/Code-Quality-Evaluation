def main():
    import sys
    input=sys.stdin.readline

    N = int(input().strip())
    presses = []
    for _ in range(N):
        x, s = input().split()
        A = int(x)
        presses.append((A, s))

    INF = 10**9
    # dp[i][l][r] = minimum fatigue after the i-th press,
    #               if the left hand is on key l and the right hand is on key r.
    dp = [[[INF]*101 for _ in range(101)] for __ in range(N+1)]

    # We can start (before any press) with our hands on any keys at zero cost.
    for l in range(1, 101):
        for r in range(1, 101):
            dp[0][l][r] = 0

    # Fill dp table
    for i in range(N):
        key, hand = presses[i]  # key to press, hand used (L or R)
        for l in range(1, 101):
            for r in range(1, 101):
                if dp[i][l][r] == INF:
                    continue
                cost_so_far = dp[i][l][r]

                if hand == 'L':
                    # Move left hand to "key", right stays at r
                    dp[i+1][key][r] = min(dp[i+1][key][r], cost_so_far + abs(key - l))
                else:
                    # Move right hand to "key", left stays at l
                    dp[i+1][l][key] = min(dp[i+1][l][key], cost_so_far + abs(key - r))

    # The answer is the minimum fatigue after all presses, over all possible final positions
    answer = INF
    for l in range(1, 101):
        for r in range(1, 101):
            answer = min(answer, dp[N][l][r])

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()