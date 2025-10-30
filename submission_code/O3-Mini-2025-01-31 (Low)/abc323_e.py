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
    
    # In our scenario, songs are chosen uniformly, so every new song is chosen with probability 1/N.
    # We want the probability that at time (X + 0.5) seconds, we are in a song and that song is song 1.
    #
    # The process:
    #   At time 0, we start the first song. When a song finishes, a new one immediately begins.
    # For a given sequence of songs, let S be the cumulative time after finishing songs.
    # Let t be the total duration (an integer) of songs before the current song.
    # Then the current song (with duration d) spans the time interval [t, t + d].
    # We want that t <= X+0.5 < t + d and the current (ongoing) song is song 1.
    #
    # Because all durations d are integers, note that X+0.5 is a half-integer.
    # The condition t <= X+0.5 < t + d is equivalent to:
    #    t <= X+0.5  and  X+0.5 - t < d.
    # Since X+0.5 - t is a half-integer, the inequality d > (X+0.5 - t) becomes
    #    d >= (X - t) + 1.
    # 
    # We compute dp[t]: the probability that the total duration of fully played songs
    # (i.e. songs that ended before time X+0.5) is exactly t.
    # We only need dp[t] for t <= X because if t > X then the song in progress would have started after time X+0.5.
    #
    # Recurrence:
    #   dp[0] = 1.
    #   For every t from 0 to X, for each song i with duration T[i]:
    #      Let next time nt = t + T[i].
    #      If nt <= X:
    #         then update dp[nt] += dp[t]*(1/N)
    #      Else:
    #         It means the chosen song will be playing at time X+0.5.
    #         We add dp[t]*(1/N) to the answer if and only if i==0 (song 1) and T[0] >= (X - t) + 1.
    
    invN = pow(N, mod - 2, mod)
    dp = [0] * (X+1)  # dp[t] is probability that songs have completed exactly at time t.
    dp[0] = 1
    ans = 0
    for t in range(X+1):
        prob = dp[t]
        if prob == 0:
            continue
        for i in range(N):
            d = T[i]
            nt = t + d
            if nt <= X:
                dp[nt] = (dp[nt] + prob * invN) % mod
            else:
                if i == 0 and d >= (X - t) + 1:
                    ans = (ans + prob * invN) % mod
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()