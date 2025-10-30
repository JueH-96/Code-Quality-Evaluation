class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        from bisect import bisect_left, bisect_right
        n = len(intervals)
        K = 4  # Maximum number of intervals to choose

        # Prepare intervals with start, end, weight, index
        intervals = [ (start, end, weight, idx) for idx, (start, end, weight) in enumerate(intervals) ]

        # Sort intervals by end time
        intervals.sort(key=lambda x: x[1])

        # Precompute p[i]: the last interval that doesn't overlap with interval i
        ends = [ interval[1] for interval in intervals ]  # list of end times
        starts = [ interval[0] for interval in intervals ]  # list of start times
        p = [ -1 ] * n
        for i in range(n):
            # Find the rightmost j where intervals[j].end < intervals[i].start
            # Since intervals are sorted by end time, we can use bisect_right on ends
            idx = bisect_left(ends, intervals[i][0]) - 1
            p[i] = idx

        # Initialize dp[i][k] = (max_weight, seq_of_indices)
        # dp[i][k] means considering intervals[0..i], selecting at most k intervals
        dp = [ [ (0, []) for _ in range(K+1) ] for _ in range(n+1) ]  # dp[0][k] = (0, [])

        for i in range(1, n+1):  # i from 1 to n
            start_i, end_i, weight_i, index_i = intervals[i-1]
            for k in range(K+1):  # k from 0 to K
                # Option1: skip interval i-1
                dp[i][k] = dp[i-1][k]

                # Option2: take interval i-1 if k > 0
                if k > 0:
                    # Find p[i-1]
                    j = p[i-1]
                    if j != -1:
                        prev_weight, prev_seq = dp[j+1][k-1]
                        current_weight = prev_weight + weight_i
                        current_seq = prev_seq + [index_i]
                    else:
                        current_weight = weight_i
                        current_seq = [index_i]

                    # Compare Option1 and Option2
                    opt1_weight, opt1_seq = dp[i][k]
                    if current_weight > opt1_weight:
                        dp[i][k] = (current_weight, current_seq)
                    elif current_weight == opt1_weight:
                        # Compare sequences lexographically
                        if current_seq < opt1_seq:
                            dp[i][k] = (current_weight, current_seq)
                        # else, keep dp[i][k] as is
                # else, dp[i][k] remains dp[i-1][k]
        
        # After filling dp table, find the best among dp[n][1..K]
        best_weight = -1
        best_seq = []
        for k in range(1, K+1):
            weight, seq = dp[n][k]
            if weight > best_weight:
                best_weight = weight
                best_seq = seq
            elif weight == best_weight:
                if seq < best_seq:
                    best_seq = seq  # Pick lex smallest seq

        # Sort the indices in ascending order
        best_seq.sort()
        return best_seq