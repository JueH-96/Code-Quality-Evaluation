def main():
    import sys, bisect
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    T = int(next(it))
    S = next(it).strip()
    positions = [int(next(it)) for _ in range(n)]
    
    # Create a list of ants as tuples: (position, direction)
    # Note: direction is a character: '0' means facing negative and '1' means facing positive.
    ants = [(positions[i], S[i]) for i in range(n)]
    # Sort the ants by their position on the number line.
    ants.sort(key=lambda x: x[0])
    
    # Create an array of just positions for binary search operations.
    posArr = [ant[0] for ant in ants]
    # Build a prefix sum array for counting how many ants (in the sorted order) face the negative direction.
    # prefix[i] is the count up to (but not including) index i.
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + (1 if ants[i][1] == '0' else 0)
    
    # Explanation:
    # Two ants, say ant i and ant j with positions X_i and X_j (with X_i < X_j) and directions d_i and d_j,
    # will pass each other if ant i goes right (d_i == '1') and ant j goes left (d_j == '0')
    # and they meet before time (T + 0.1).
    # Meeting time, if they go in opposite directions, is (X_j - X_i) / 2.
    # So we need (X_j - X_i) / 2 <= T + 0.1.
    # Since positions are integers, the effective condition becomes X_j - X_i <= floor(2*(T+0.1)).
    # Note: 2*(T+0.1) = 2T + 0.2. Its floor is 2T since 0.2 does not push to the next integer.
    #
    # Thus, for every ant i facing right (d_i == '1'), we want to count the number of ants j (with j > i)
    # that face left (d_j == '0') and satisfy X_j - X_i <= 2T.
    
    total_pairs = 0
    for i in range(n):
        # Consider only ants moving to the right.
        if ants[i][1] == '1':
            # Determine the furthest position this ant i can meet an ant coming from the left
            # within time (T+0.1). That maximum allowed position is ants[i][0] + 2T.
            limit = ants[i][0] + 2 * T
            # Use binary search to find the rightmost index where position is <= limit.
            # bisect_right returns the first index with a value greater than limit.
            hi = bisect.bisect_right(posArr, limit) - 1
            # We only care about ants with j > i.
            if hi > i:
                # Using the prefix sum array, count how many ants in indices (i+1, hi] face left.
                count_left = prefix[hi+1] - prefix[i+1]
                total_pairs += count_left

    sys.stdout.write(str(total_pairs))

if __name__ == "__main__":
    main()