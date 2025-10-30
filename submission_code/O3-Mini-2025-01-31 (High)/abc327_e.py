def main():
    import sys, math
    import numpy as np
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    # read performance scores and convert to float for arithmetic precision
    perf = list(map(float, data[1:]))

    # We use dp[k] to represent the maximum weighted sum S of a subsequence
    # of length k. The weighted sum is defined recursively:
    #   S(new) = 0.9 * S(old) + P
    # where extending a subsequence of length k (with weighted sum S(old))
    # by contest performance P gives a subsequence of length k+1.
    #
    # Note that if the chosen contests are Q_1, Q_2, ..., Q_k, in order,
    # then the weighted sum S is:
    #    S = (0.9)^(k-1)*Q_1 + (0.9)^(k-2)*Q_2 + ... + 1*Q_k,
    # and the denominator in the rating formula is:
    #    D = Σ_{j=0}^{k-1} (0.9)^j = (1-0.9^k)/(1-0.9) = 10*(1-0.9^k).
    #
    # The final rating for a subsequence of length k is:
    #    R = (S / D) - (1200/√k).
    
    # dp[0] represents the “empty sequence” with value 0.
    # We only allow sequences of length at least 1 for the final answer.
    dp = np.full(n + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    # curr_max is the maximum subsequence length achieved so far.
    curr_max = 0

    # Process contests one at a time in the given order.
    for p in perf:
        # For each already-achieved subsequence (of length 0 to curr_max),
        # consider extending it with the current contest.
        # For a subsequence of length k with weighted sum dp[k], appending p gives:
        #   new S = 0.9 * dp[k] + p
        # This new value becomes a candidate for dp[k+1].
        # We use a vectorized slice: dp[0:curr_max+1] holds the previous states.
        cand = 0.9 * dp[:curr_max + 1] + p
        # The candidate for a sequence of length (i+1) is stored at dp[i+1].
        # We update these positions taking the maximum over all possible ways.
        dp[1:curr_max + 2] = np.maximum(dp[1:curr_max + 2], cand)
        # If we got a valid new subsequence of length curr_max+1, update the count.
        if dp[curr_max + 1] != -np.inf:
            curr_max += 1

    # Now compute the rating for each subsequence length l (from 1 to curr_max).
    # For a subsequence of length l:
    #   weighted average = dp[l] / (10*(1 - 0.9^l))
    #   rating R = weighted average - 1200/√l
    best_rating = -np.inf
    for l in range(1, curr_max + 1):
        denom = 10 * (1 - (0.9 ** l))  # geometric sum: 1+0.9+...+0.9^(l-1)
        rating = dp[l] / denom - 1200 / math.sqrt(l)
        if rating > best_rating:
            best_rating = rating

    sys.stdout.write(str(best_rating))
    
    
if __name__ == '__main__':
    main()