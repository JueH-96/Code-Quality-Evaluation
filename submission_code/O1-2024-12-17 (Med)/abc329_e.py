def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    T = data[3]

    # We use a dynamic-programming approach based on the idea that:
    # dp[i] = True means "it is possible to finalize (fix) all positions < i
    # so that they match S[0..i-1] by some sequence of operations."
    #
    # From dp[i] = True, we try to "cover" further positions by choosing
    # an operation (i.e. placing T somewhere) that could finalize position i
    # and possibly more positions to the right, until we hit a mismatch.
    #
    # The key observation is:
    #  - In the final arrangement, each position is ultimately set by exactly
    #    one "last" coverage operation of length M.
    #  - Operations can overwrite each other, so earlier ones do not impose
    #    constraints except for positions that do not get overwritten again.
    #  - Hence, from dp[i], one valid "move" is to pick a start s in
    #    [i - (M - 1), i], clamped to [0..(N-M)], and then see how many
    #    consecutive positions from i onward can match T in that sliding window.
    #    Those positions can be considered finalized by this coverage. We
    #    update dp for all those newly finalized positions j+1.
    #
    # Since M <= 5, the inner loop of length M is not too large, and this
    # approach remains O(N * M^2) = O(25*N), which is acceptable for N up
    # to ~2*10^5 in a reasonably efficient implementation.

    dp = [False] * (N + 1)
    dp[0] = True  # No characters fixed yet is trivially possible

    # For faster checks, we'll just do direct comparisons since M is small.
    for i in range(N):
        if not dp[i]:
            continue
        # From position i, try all possible starts s of the coverage that can fix i
        # s in [max(0, i - (M-1)) .. min(i, N-M)]
        start_min = i - (M - 1)
        if start_min < 0:
            start_min = 0
        start_max = i
        if start_max > N - M:
            start_max = N - M
        for s in range(start_min, start_max + 1):
            # The coverage is [s..s+M-1].
            # We'll match consecutive positions j from i.. up to s+M-1
            # as long as T[j-s] == S[j].
            end = s + M - 1
            j = i
            while j <= end and S[j] == T[j - s]:
                dp[j + 1] = True
                j += 1
                # once j goes beyond end or mismatch occurs, we stop

    print("Yes" if dp[N] else "No")

# Don't forget to call main():
if __name__ == "__main__":
    main()