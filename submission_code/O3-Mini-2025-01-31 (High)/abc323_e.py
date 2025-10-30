def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    T = [int(next(it)) for _ in range(N)]
    # For probability computations modulo mod,
    # we will work with probabilities multiplied by the inverse of N repeatedly.
    invN = pow(N, mod - 2, mod)
    
    # dp[s] will represent the probability (mod mod) that a song started exactly at time s
    # (i.e. after finishing a sequence of songs, the next song begins at time s)
    # where s is an integer in [0, X]. 
    # Later we will “overshoot” the time X by selecting a song that does not finish by time X,
    # and if that song is song1 we add its probability to the answer.
    dp = [0] * (X + 1)
    dp[0] = 1  # we start at time 0
    ans = 0
    T1 = T[0]
    
    # For songs with index != 1, group them by duration.
    freq = {}
    for j in range(1, N):
        d = T[j]
        freq[d] = freq.get(d, 0) + 1

    # For each possible current renewal time s (a time when a new song started),
    # extend the DP by choosing the next song uniformly among the N songs.
    # If the chosen song finishes by time X (i.e. s + duration <= X),
    # then update dp[s + duration] with that probability.
    # Otherwise, the song is in progress at time X+0.5.
    # Since the problem asks for the probability that at time X + 0.5 the current song is song1,
    # we only add a contribution to our answer when the overshooting song is song1.
    for s in range(X + 1):
        cur = dp[s]
        if cur == 0:
            continue
        # Transition if we choose song1:
        ns = s + T1
        if ns <= X:
            dp[ns] = (dp[ns] + cur * invN) % mod
        else:
            # This chosen song overshoots time X.
            # Note that the playing interval is from s to s+T1,
            # and since X+0.5 lies strictly between (s, s+T1), we add the probability.
            ans = (ans + cur * invN) % mod
        # Transition for all other songs:
        for d, cnt in freq.items():
            ns = s + d
            if ns <= X:
                dp[ns] = (dp[ns] + cur * cnt * invN) % mod
            # If ns > X, then even though the song overshoots time X,
            # it does not count toward our answer because it is not song1.
            # So we do nothing in that case.
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()