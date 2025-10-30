def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    # dp[k] will represent the maximum weighted sum we can obtain for a subsequence of length k.
    # When we pick a contest, if the current subsequence is of length k with weighted sum S,
    # then adding a new contest with performance Q gives a new weighted sum:
    # new_S = S * 0.9 + Q
    # because every previously chosen contest gets discounted by a factor of 0.9 and the new contest
    # gets a weight of 1.
    #
    # Our eventual rating for a subsequence of length k having weighted sum S is:
    # R = (S / W) - (1200 / sqrt(k))
    # where W = sum_{i=1}^{k} (0.9)^(k-i). It can be computed as:
    # W = (1 - 0.9^k) / 0.1.
    #
    # We need to choose at least one contest.
    
    INF = -10**18
    dp = [INF] * (N + 1)
    dp[0] = 0.0  # Starting with an empty subsequence
    best = -10**18
    
    # Precompute 0.9 powers to quickly compute the weight sum W for any length k.
    pow09 = [1.0] * (N + 2)
    for i in range(1, N + 2):
        pow09[i] = pow09[i - 1] * 0.9
    
    # We'll process contests in the order they were held.
    max_length = 0
    for p in P:
        # Traverse in reverse order to avoid using updated values in the same round.
        for k in range(max_length, -1, -1):
            # If a subsequence of length k exists, we can try to extend it with performance p.
            if dp[k] == INF:
                continue
            new_sum = dp[k] * 0.9 + p
            # dp[k+1] is updated to the best weighted sum possible for a subsequence of length k+1.
            if new_sum > dp[k + 1]:
                dp[k + 1] = new_sum
            # Now, compute the rating for this subsequence.
            length = k + 1
            weight_sum = (1.0 - pow09[length]) / 0.1
            rating = (new_sum / weight_sum) - (1200.0 / math.sqrt(length))
            if rating > best:
                best = rating
        if max_length < N:
            max_length += 1

    sys.stdout.write("{:.15f}".format(best))

if __name__ == "__main__":
    main()