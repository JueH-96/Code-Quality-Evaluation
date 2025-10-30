def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    data = sys.stdin.read().split()
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:]))

    invN = pow(N, MOD-2, MOD)

    # dp[s] = probability (mod) that exactly s seconds of songs have just completed
    # before the current (ongoing) song starts.
    dp = [0] * (X+1)
    dp[0] = 1  # with probability 1 we start at time 0 with no completed songs

    for s in range(1, X+1):
        val = 0
        # consider that the last completed song had length T_i
        for t in T:
            if t <= s:
                val += dp[s-t]
        dp[s] = val % MOD
        # multiply by 1/N to account for the uniform choice of the last song
        dp[s] = dp[s] * invN % MOD

    # Now sum over all s so that s <= X < s + T[0], i.e. s >= X - T1 + 1
    T1 = T[0]
    start = max(0, X - T1 + 1)
    ans = 0
    for s in range(start, X+1):
        # probability prefix ends at s AND next song is song 1
        ans = (ans + dp[s] * invN) % MOD

    print(ans)

if __name__ == "__main__":
    main()