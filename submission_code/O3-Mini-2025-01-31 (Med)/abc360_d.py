def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    T = int(next(it))
    S = next(it)
    X = [int(next(it)) for _ in range(n)]
    
    # Create a list of (position, direction) tuples.
    # Direction is kept as a character from S: '0' for left, '1' for right.
    ants = [(X[i], S[i]) for i in range(n)]
    
    # Sort the ants by position so that for any pair (i,j) with i<j, we have X_i < X_j.
    ants.sort(key=lambda x: x[0])
    positions = [ant[0] for ant in ants]
    
    # We need to count pairs (i, j) such that ant i is moving right ('1'),
    # ant j is moving left ('0'), and they meet during time strictly before (T+0.1).
    # They meet when (X_j - X_i) / 2 < T+0.1, i.e.
    # X_j - X_i < 2T + 0.2.
    # Since X_j - X_i is an integer, this condition becomes X_j - X_i <= 2T.
    
    # To efficiently count for every ant moving right, we precompute a prefix sum
    # for the number of ants moving left ('0') in the sorted list.
    prefix_left = [0] * (n + 1)
    for i in range(n):
        prefix_left[i+1] = prefix_left[i] + (1 if ants[i][1] == '0' else 0)
        
    total_pairs = 0
    
    # For every ant moving right, count the ants to its right that are moving left
    # and are within a distance of 2T.
    for i in range(n):
        pos, d = ants[i]
        if d == '1':  # ant moving right
            # The condition for potential meeting is X_j <= pos + 2T.
            # Use bisect_right to find the first index where the position exceeds pos + 2T.
            j = bisect.bisect_right(positions, pos + 2 * T)
            # Only consider ants that come after ant i, i.e., indices from i+1 to j-1.
            if j > i+1:
                count_left = prefix_left[j] - prefix_left[i+1]
                total_pairs += count_left

    sys.stdout.write(str(total_pairs))

if __name__ == '__main__':
    main()