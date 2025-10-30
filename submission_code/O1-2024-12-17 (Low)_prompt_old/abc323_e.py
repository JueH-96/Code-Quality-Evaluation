def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = int(input_data[1])
    T = list(map(int, input_data[2:]))

    MOD = 998244353
    # Modular inverse of n under MOD (Fermat's little theorem, since MOD is prime)
    invN = pow(N, MOD - 2, MOD)

    # We'll call T_1 the duration of the first song (index 0 in the list)
    T1 = T[0]

    # dp[t] = probability (mod MOD) that a new song starts exactly at time t
    # We only need to track t up to X (since X+0.5 can't be inside a new song
    # that starts beyond X for integral start times).
    dp = [0] * (X + 1)
    dp[0] = 1  # At time 0, we definitely start a song.

    for t in range(X + 1):
        if dp[t] == 0:
            continue
        # Distribute this probability equally among all N songs
        ways = dp[t]
        distribute = (ways * invN) % MOD
        for length in T:
            nt = t + length
            if nt <= X:
                dp[nt] = (dp[nt] + distribute) % MOD

    # Probability that song 1 is playing at time (X + 0.5)
    # If a song starts at time t (an integer), it covers t <= t' < t + length.
    # We want t <= X+0.5 < t + T1 => t <= X and t + T1 > X+0.5 => t + T1 >= X+1
    # => t >= X+1 - T1. Also the chance that the next chosen song is song 1 is (1/N).
    start_min = max(0, X + 1 - T1)
    ans = 0
    for t in range(start_min, X + 1):
        ans = (ans + dp[t]) % MOD

    # Multiply by 1/N
    ans = (ans * invN) % MOD

    print(ans)