def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    H = [int(next(it)) for _ in range(n)]
    
    # Group positions by height.
    positions_by_height = {}
    for index, height in enumerate(H):
        positions_by_height.setdefault(height, []).append(index)
    
    best = 1  # at least one building can be chosen.
    
    # For each height, we check the maximum number of indices that form an arithmetic progression.
    for height, positions in positions_by_height.items():
        count = len(positions)
        if count <= best:
            # if even all buildings of this height are less than or equal to current best,
            # no need to check further for AP.
            continue
        if count == 1:
            best = max(best, 1)
            continue
        
        # dp[j] will be a dictionary mapping difference d to the longest AP ending at positions[j]
        dp = [dict() for _ in range(count)]
        local_best = 1
        for j in range(count):
            for i in range(j):
                d = positions[j] - positions[i]
                # If there's already an arithmetic sequence ending at index i with difference d,
                # we extend that sequence by the current position.
                # Otherwise, we start a new sequence of two items.
                curr_length = dp[i].get(d, 1) + 1
                # Update the sequence ending at j with difference d.
                dp[j][d] = max(dp[j].get(d, 0), curr_length)
                local_best = max(local_best, dp[j][d])
        best = max(best, local_best)
    
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()