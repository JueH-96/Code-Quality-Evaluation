def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N, X = map(int, input().split())
    T = list(map(int, input().split()))

    # modular inverse of N
    invN = pow(N, MOD - 2, MOD)

    # dp[l] = probability (mod MOD) that some song-boundary occurs exactly at time l
    dp = [0] * (X + 1)
    dp[0] = 1  # with probability 1 we start at time 0

    # only durations <= X can contribute to dp[1..X]
    durations = sorted(t for t in T if t <= X)

    for l in range(1, X + 1):
        s = 0
        # sum dp[l - t] over all song-lengths t that fit
        for t in durations:
            if t > l:
                break
            s += dp[l - t]
        # reduce mod and multiply by 1/N
        if s >= MOD:
            s %= MOD
        dp[l] = s * invN % MOD

    # We want the probability that song 1 is playing at time X + 0.5
    # That equals (1/N) * sum_{L = max(0, X+1-T1)}^{X} dp[L]
    T1 = T[0]
    L0 = X + 1 - T1
    if L0 < 0:
        L0 = 0
    # Sum dp[L0..X]
    total = sum(dp[L0:]) % MOD
    ans = total * invN % MOD

    print(ans)

if __name__ == "__main__":
    main()