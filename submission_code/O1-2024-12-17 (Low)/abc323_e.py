def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = int(input_data[1])
    T = list(map(int, input_data[2:]))

    MOD = 998244353
    
    # Precompute modular inverse of N
    invN = pow(N, MOD - 2, MOD)

    # Frequency array of song lengths (up to 10000 as per constraints)
    # freq[t] = how many songs have length t
    maxT = 0
    freq = [0]*(10001)
    for length in T:
        freq[length] += 1
        if length > maxT:
            maxT = length
    
    # We only need to compute probabilities dp up to X + maxT
    # dp[s] = probability (mod MOD) that "a new song starts exactly at time s"
    limit = X + maxT
    dp = [0]*(limit+1)
    dp[0] = 1  # at time 0, a new song (the very first) starts with probability 1

    # Propagate probabilities
    # For each time s where a song starts, distribute probability of starting the next song at s + t
    # for each possible length t (according to freq[t]) with probability freq[t]/N
    for s in range(limit+1):
        val = dp[s]
        if val == 0:
            continue
        # multiply by 1/N (mod) to get probability share for each next length
        share = (val * invN) % MOD
        # add to dp[s + t] for each t
        # dp[s+t] += share * freq[t]
        # 1 <= t <= maxT
        # but only up to limit
        i_max = limit - s
        # Instead of looping over all t up to maxT (which can be 10000) unconditionally,
        # do it but break if s+t>limit.
        # This is still an O(limit*maxT) approach, potentially up to ~200 million operations.
        # We'll implement it carefully; hopefully it passes under the test environment.
        add_to = dp
        for t_len in range(1, maxT+1):
            if t_len > i_max:
                break
            f = freq[t_len]
            if f != 0:
                nxt = s + t_len
                # update dp[nxt] = (dp[nxt] + share * f) % MOD
                inc = share * f
                inc_mod = inc % MOD
                add_to[nxt] = (add_to[nxt] + inc_mod) % MOD

    # We want the probability that song 1 (whose length = T[0]) is playing at time X + 0.5.
    # That means the start time s of that song is in [ X+0.5 - T1, X+0.5 ), and specifically
    # the chosen song was #1 (which has probability 1/N), and T1 > (X+0.5 - s).
    # Because s is integer, we set:
    #   s_min = ceil(X+0.5 - T1)
    #   s_max = floor(X+0.5) = X
    #
    # But we only sum for s in [0 .. X], and also s+T1 > X+0.5 => s >= X+0.5 - T1
    # => s >= floor(X+0.5 - T1) + 1
    # We'll simplify with integer arithmetic:
    #   s >= (X - T1 + 0.5) => s >= (X - T1) if we think in floors, carefully:
    # Let's pick s_min = max(0, X - T[0] + 1).
    # Then we sum dp[s] * (1/N) for s in [s_min .. X].
    
    T1 = T[0]
    # If s+T1 > X+0.5 => s > X+0.5 - T1 => s >= floor(X+0.5 - T1) +1
    # floor(X+0.5) = X
    # so s_min = max(0, X - T1 +1)
    s_min = X - T1 + 1
    if s_min < 0:
        s_min = 0
    if s_min > X:
        # no valid start time
        print(0)
        return
    
    # sum up dp[s] * (1/N) for s = s_min..X
    ans = 0
    for s in range(s_min, X+1):
        ans = (ans + dp[s]) % MOD
    
    # multiply by 1/N
    ans = (ans * invN) % MOD
    
    print(ans)