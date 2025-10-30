def main():
    import sys
    mod = 998244353
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    T = [int(next(it)) for _ in range(N)]
    
    # Explanation:
    # We have an infinite random sequence of songs. At each new song, the process:
    #   - Starts at an integer time (since all songs have integer lengths).
    #   - Chooses one song uniformly (probability 1/N each).
    #   - Plays it entirely.
    # We want the probability that at time (X+0.5) seconds, the song playing is song 1.
    #
    # Notice: Since song boundaries are at integer seconds, X+0.5 lies strictly inside some song.
    # For the song that starts at time s (with s an integer, s <= X) and has duration t,
    # the song covers time (X+0.5) if and only if:
    #     s <= X+0.5 < s + t.
    # Because s is integer, s <= X+0.5 is equivalent to s <= X.
    # And since s+t is an integer, the inequality X+0.5 < s+t is equivalent to s+t >= X+1.
    # So, if the song chosen from the boundary at time s is song1, we need s + T1 >= X+1.
    #
    # Approach:
    # 1. Let dp[s] be the probability that the process has finished a whole number
    #    of songs and the total elapsed time is exactly s seconds (for s in [0, X]).
    #    We start with dp[0] = 1.
    # 2. For each such state s (0 <= s <= X), we try to pick a next song.
    #    - If s + T_i <= X, we add dp[s]*(1/N) to dp[s + T_i], since the process continues.
    #    - If s + T_i > X, the next song is the "final" song covering time X+0.5.
    #      In that case, if i is 0 (song 1, using 0-based indexing) and it satisfies
    #      s + T1 >= X+1, then we add dp[s]*(1/N) to our answer.
    #
    # The answer is the total probability mod 998244353.
    
    dp = [0] * (X + 1)
    dp[0] = 1
    ans = 0
    invN = pow(N, mod-2, mod)  # modular inverse of N
    
    for s in range(X + 1):
        if dp[s] == 0:
            continue
        current_prob = dp[s]
        for idx, t in enumerate(T):
            ns = s + t
            if ns <= X:
                dp[ns] = (dp[ns] + current_prob * invN) % mod
            else:
                # This song will be in play at time X+0.5 if s <= X and 
                # s + t >= X+1. By our setup, when ns > X, we check:
                if idx == 0 and t >= (X + 1 - s):
                    ans = (ans + current_prob * invN) % mod
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()