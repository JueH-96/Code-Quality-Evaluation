def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    # performances; convert to float
    P = list(map(float, data[1:]))
    
    # dp[k] will store the maximum weighted sum obtainable when a subsequence of length k is chosen.
    # We will use 1-indexing for k so dp[1...N] are meaningful.
    dp = [float("-inf")] * (N+1)
    # current_max keeps track of the maximum sequence length achieved so far.
    current_max = 0
    
    # Process contests one by one in order.
    for x in P:
        # Extend every subsequence of length k (largest k first so we don’t use updated values in the same iteration):
        for k in range(current_max, 0, -1):
            new_val = dp[k] * 0.9 + x
            if new_val > dp[k+1]:
                dp[k+1] = new_val
        # Also, you can start a new sequence (of length 1) with the current contest:
        if x > dp[1]:
            dp[1] = x
        # Since we have processed one more contest, our maximum subsequence length can increase by one.
        current_max += 1
        if current_max > N:
            current_max = N

    best = -10**18
    # Evaluate the rating for every possible sequence length k = 1,2,...,current_max.
    for k in range(1, current_max+1):
        # the denominator is: 10*(1 - (0.9)^k)
        denom = 10 * (1 - (0.9) ** k)
        avg = dp[k] / denom
        rating = avg - 1200 / math.sqrt(k)
        if rating > best:
            best = rating

    # Print the result with at least 10^-6 precision.
    sys.stdout.write("{:.15f}".format(best))
    
if __name__ == "__main__":
    main()