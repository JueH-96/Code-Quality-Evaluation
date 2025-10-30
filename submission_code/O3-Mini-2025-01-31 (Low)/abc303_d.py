def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3].strip()

    n = len(S)
    INF = 10**18

    # dp[i][state]: Minimum cost to produce first i characters finishing in state.
    # state = 0 for Caps Lock off, state = 1 for Caps Lock on.
    dp = [[INF, INF] for _ in range(n + 1)]
    
    # Initially, the CapsLock is off (0 cost) but we can also toggle immediately.
    dp[0][0] = 0
    dp[0][1] = Z  # toggling to on

    # For each character, we want to decide in which state we should produce the next letter.
    # Pressing keys:
    #   - Press only 'a':
    #         if state==0 (off) then produces 'a'
    #         if state==1 (on) then produces 'A'
    #        Cost = X milliseconds.
    #   - Press Shift+'a':
    #         if state==0 then produces 'A'
    #         if state==1 then produces 'a'
    #        Cost = Y milliseconds.
    # Toggling CapsLock (# state flip) costs Z milliseconds.
    #
    # We'll keep the dp and allow state toggles at no letter-production,
    # meaning that before processing any letter, we relax our dp values via toggle.
    for i in range(n):
        # Relax dp[i] by toggling if beneficial.
        dp[i][0] = min(dp[i][0], dp[i][1] + Z)
        dp[i][1] = min(dp[i][1], dp[i][0] + Z)

        for state in (0, 1):
            cur_cost = dp[i][state]
            # Determine the result if we press 'a' (without Shift):
            #   if state == 0 then it contributes 'a'
            #   if state == 1 then it contributes 'A'
            c_normal = 'a' if state == 0 else 'A'
            if S[i] == c_normal:
                if cur_cost + X < dp[i + 1][state]:
                    dp[i + 1][state] = cur_cost + X

            # Determine the result if we press 'a' with Shift:
            #   if state == 0 then it contributes 'A'
            #   if state == 1 then it contributes 'a'
            c_shift = 'A' if state == 0 else 'a'
            if S[i] == c_shift:
                if cur_cost + Y < dp[i + 1][state]:
                    dp[i + 1][state] = cur_cost + Y

    # After processing all letters, relax toggling one more time.
    dp[n][0] = min(dp[n][0], dp[n][1] + Z)
    dp[n][1] = min(dp[n][1], dp[n][0] + Z)
    
    result = min(dp[n][0], dp[n][1])
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()