def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    # Read input values
    N = int(next(it))
    M = int(next(it))
    black = [int(next(it)) for _ in range(N)]
    white = [int(next(it)) for _ in range(M)]
    
    # Sort both lists in descending order.
    # For black balls, picking as many with high positive values increases the sum.
    # For white balls, we only want to pick the ones with higher values if they help the total sum.
    black.sort(reverse=True)
    white.sort(reverse=True)
    
    # Compute prefix sums for black.
    # prefixB[i] is the total of the first i black balls (i from 0 to N)
    prefixB = [0] * (N + 1)
    for i in range(1, N + 1):
        prefixB[i] = prefixB[i - 1] + black[i - 1]
    
    # We must choose at least as many black balls as white balls.
    # When we decide to choose k white balls (from the sorted white list),
    # we must pick at least k black balls. However, after picking k black balls, we can
    # additionally pick any extra black balls if they increase the sum.
    # Hence, given the prefix sums of the black balls, for each k we want the maximum sum possible
    # by choosing any r black balls with r >= k.
    best_from = [0] * (N + 2)  # best_from[i] = max(prefixB[i], prefixB[i+1], ... prefixB[N])
    best_from[N] = prefixB[N]
    best_from[N + 1] = -10**18  # Set a very low value for boundary consistency.
    for i in range(N - 1, -1, -1):
        # Choose the better between taking exactly i balls, or a better option from i+1 onward.
        best_from[i] = prefixB[i] if prefixB[i] >= best_from[i + 1] else best_from[i + 1]
    
    # Compute prefix sums for white.
    prefixW = [0] * (M + 1)
    for i in range(1, M + 1):
        prefixW[i] = prefixW[i - 1] + white[i - 1]
    
    # Try all possible numbers k (0 <= k <= min(M, N)).
    # k represents the number of white balls chosen.
    # For each k, we are forced to pick at least k black balls, so the maximum additional black sum is best_from[k].
    # Note that choosing no balls (k = 0) is always an option.
    ans = 0
    max_k = min(M, N)
    for k in range(0, max_k + 1):
        current_sum = best_from[k] + prefixW[k]
        if current_sum > ans:
            ans = current_sum
    
    # Print the maximum possible sum.
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()