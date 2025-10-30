def main():
    import sys
    from bisect import bisect_right
    
    data = sys.stdin.read().strip().split()
    # Parsing input.
    it = iter(data)
    N = int(next(it))
    T = int(next(it))
    S = next(it).strip()
    X = [int(next(it)) for _ in range(N)]
    
    # We consider an ant with direction "1" as moving right and "0" as moving left.
    # Two ants pass (collide) if a right–moving ant and a left–moving ant meet.
    # Suppose we have two ants with positions x_i and x_j (with x_i < x_j). If ant i moves right and ant j moves left,
    # they meet at time t such that:
    #     x_i + t = x_j - t   ==>   t = (x_j - x_i) / 2.
    # They pass each other (before or exactly at time T+0.1) if:
    #     (x_j - x_i) / 2 <= T+0.1  ->  x_j - x_i <= 2T + 0.2.
    # Since x_j - x_i is an integer and 2T+0.2 is not an integer, 
    # the effective condition becomes: x_j - x_i <= 2T.
    
    # Create a list of ants with (position, is_moving_right) where is_moving_right is True if the ant
    # is facing positive (S = "1") and False otherwise.
    ants = [(X[i], S[i] == '1') for i in range(N)]
    ants.sort(key=lambda x: x[0])
    
    # Create a list of positions for binary search.
    positions = [ant[0] for ant in ants]
    
    # We are only interested in collisions where the left ant (in positions order)
    # is moving right and the right ant is moving left.
    # To do this efficiently, we build an indicator and prefix sum for ants moving left.
    left_indicator = [0] * N  # 1 if ant is moving left, else 0.
    for i, (pos, is_right) in enumerate(ants):
        if not is_right:
            left_indicator[i] = 1
    prefix = [0] * N
    prefix[0] = left_indicator[0]
    for i in range(1, N):
        prefix[i] = prefix[i - 1] + left_indicator[i]
    
    # Helper function to quickly get the sum in a range [l, r] in the prefix sum array.
    def range_sum(l, r):
        if l > r:
            return 0
        return prefix[r] if l == 0 else prefix[r] - prefix[l - 1]
    
    ans = 0
    max_gap = 2 * T  # As explained above.
    
    # For each ant that is moving right (i.e. is_right == True), find how many ants to its right (with larger positions)
    # are moving left and are within a gap of max_gap. They satisfy the condition for collision.
    for i in range(N):
        pos, is_right = ants[i]
        if not is_right:
            continue
        # Using binary search, find the last index "hi" such that positions[hi] is <= pos + max_gap.
        hi = bisect_right(positions, pos + max_gap) - 1
        if hi <= i:
            continue
        # Query the number of ants moving left in the index range [i+1, hi].
        count_collisions = range_sum(i + 1, hi)
        ans += count_collisions
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()