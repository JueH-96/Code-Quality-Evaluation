def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    # Read heights into a list (using 0-indexing).
    H = list(map(int, data[1:]))

    # We first compute the "next greater" index for each building.
    # For an index i (0-indexed), next_greater[i] is the smallest index j > i such that H[j] > H[i].
    # If no such j exists, we set next_greater[i] = -1.
    next_greater = [-1] * n
    stack = []
    # Process from rightmost to leftmost.
    for i in range(n - 1, -1, -1):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        if stack:
            next_greater[i] = stack[-1]
        else:
            next_greater[i] = -1
        stack.append(i)

    # The key observation:
    # For a fixed building i (the observer), when scanning from i+1 to the right,
    # the buildings that are visible are exactly those that form a chain of record–breakers.
    # In other words, if we start at building i+1 and then repeatedly jump to the next building
    # that is taller than our current one, each such building is visible.
    #
    # Thus we define a DP array (dp) for the chain length:
    # For an index i, dp[i] equals 1 (counting building i itself) plus the chain
    # from next_greater[i] if it exists. Formally:
    #    dp[i] = 1                  if next_greater[i] == -1,
    #    dp[i] = 1 + dp[next_greater[i]]   otherwise.
    # (Note: We use dp only for indices from 0 to n-1; for an observer building i, the visible count
    #  is the chain starting at position i+1.)
    dp = [0] * n
    dp[n - 1] = 1  # The chain starting at the last index: only itself.
    for i in range(n - 2, -1, -1):
        j = next_greater[i]
        if j == -1:
            dp[i] = 1
        else:
            dp[i] = 1 + dp[j]

    # Now, for each building i (0-indexed), the problem requires counting j (with i < j)
    # such that the sightline condition holds. That turns out to be exactly the length
    # of the record chain starting at building i+1.
    # If i is the last building then answer is 0.
    result = [0] * n
    for i in range(n - 1):
        result[i] = dp[i + 1]
    result[n - 1] = 0

    sys.stdout.write(" ".join(map(str, result)))

if __name__ == '__main__':
    main()