class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        """
        We have disjoint "segments" [l_i, r_i], each position in that segment has c_i coins.
        We want to find the maximum total coins one can get by picking k consecutive bags (positions).

        Because l_i, r_i, and k can be as large as 10^9, we cannot simply build a huge array.
        Instead, we note that the coin distribution on the line is piecewise-constant in disjoint intervals.
        
        Outline:
          1) Build a "difference array" of (position, delta) to represent where coin counts start or stop.
             For each interval [l, r] with coin-count c:
                - At position l, we add +c to the running coin count.
                - At position r+1, we add -c (ends that count after r).
          2) Sort these "changes" by position and build a list of segments = [(start, end, coin_value)],
             each segment meaning "from start..end inclusive, we have coin_value coins per position".
             Gaps (where coin_value = 0) are simply ignored (they do not appear in the segments list).
          3) Collect all critical boundaries (the starts and ends of the segments).  
             The maximum sub-interval of length k can begin at any of these boundaries or end at them,
             since the total changes only when the window crosses a boundary.
          4) For each boundary b in sorted order (and within the overall min..max range):
                - Compute the sum of coins in [b, b + k - 1].
                - Keep track of the maximum such sum.
          5) To compute the sum in [L, R] quickly, we use binary searches and prefix sums over segments.
             - We find which segments intersect [L, R].  
             - Sum partial coverage in the first and last intersecting segments, and
               sum fully-covered segments in-between by a prefix-sum lookup.
        
        This yields an O(N log N) solution, which is feasible for N up to 10^5.
        """

        import sys
        import bisect
        
        # 1) Build the "difference array" of changes.
        changes = []
        for l, r, c in coins:
            # Start adding c at position l
            changes.append((l, c))
            # Stop adding c after r (so at r+1 we add -c)
            changes.append((r + 1, -c))
        
        # Sort changes by the position
        changes.sort(key=lambda x: x[0])
        
        # 2) Sweep through changes to build segments
        segments = []  # Each will be (start, end, coin_value)
        running = 0    # Current running coin count
        # We'll go through changes[i], and the interval between changes[i] and changes[i+1] (exclusive of the latter) is constant
        for i in range(len(changes) - 1):
            pos, delta = changes[i]
            running += delta
            next_pos = changes[i + 1][0]
            # If running > 0, from pos..(next_pos - 1) we have 'running' coins
            if running != 0:
                # next_pos - 1 is the last integer before changes[i+1][0]
                start = pos
                end = next_pos - 1
                if start <= end:  # valid non-empty range
                    segments.append((start, end, running))
        
        # If no segments (all zero?), answer is 0
        if not segments:
            return 0
        
        # Sort segments by their start (they should already be in ascending order by construction)
        # but just to be sure:
        segments.sort(key=lambda x: x[0])
        
        # 3) Build prefix sums over these segments for fast range-sum queries
        #    prefixSum[i] = total coins from segment[0] up to segment[i-1] (inclusive).
        #    So prefixSum[0] = 0, prefixSum[i+1] = prefixSum[i] + (length_of_segment_i * c_i).
        n = len(segments)
        prefixSum = [0] * (n + 1)
        for i in range(n):
            (s, e, c) = segments[i]
            length = e - s + 1
            prefixSum[i + 1] = prefixSum[i] + length * c
        
        # We'll also keep arrays of just starts[] and ends[] so we can binary-search quickly.
        starts = [seg[0] for seg in segments]
        ends   = [seg[1] for seg in segments]
        
        # Collect boundaries = set of all segment starts and ends.
        boundarySet = set()
        for (s, e, c) in segments:
            boundarySet.add(s)
            boundarySet.add(e)
        
        # Sort boundaries
        boundaries = sorted(boundarySet)
        
        # We only need to consider sub-interval starts in [minL, maxR],
        # because starting further left or further right would not improve the result.
        minL = boundaries[0]
        maxR = boundaries[-1]

        # Helper to compute sum of coins in [L, R] using prefix sums + partial coverage
        def get_sum(L, R):
            if R < L:
                return 0
            # Find the first segment that might intersect L:
            # i = smallest index where ends[i] >= L
            i = bisect.bisect_left(ends, L)
            if i == n:
                # All segments end before L, no coverage
                return 0
            
            # Find the last segment that might intersect R:
            # j = largest index where starts[j] <= R
            j = bisect.bisect_right(starts, R) - 1
            if j < 0:
                # All segments start after R, no coverage
                return 0
            if i > j:
                # No actual overlap
                return 0
            
            total = 0
            # Partial coverage of the i-th segment (if it truly overlaps)
            si, ei, ci = segments[i]
            if ei >= L:  # it intersects
                leftBound = max(L, si)
                rightBound = min(R, ei)
                if leftBound <= rightBound:
                    total += (rightBound - leftBound + 1) * ci
            
            # If there's more than one segment in range, handle partial coverage of j-th segment
            if j > i:
                sj, ej, cj = segments[j]
                leftBound = max(L, sj)
                rightBound = min(R, ej)
                if leftBound <= rightBound:
                    total += (rightBound - leftBound + 1) * cj
            
            # Now add all fully-covered segments between (i+1) and (j-1)
            # prefix sums: sum of segments i+1..(j-1) inclusive is prefixSum[j] - prefixSum[i+1]
            if j - 1 >= i + 1:
                total += prefixSum[j] - prefixSum[i + 1]
            
            return total
        
        ans = 0
        for b in boundaries:
            if b > maxR:
                break
            L = b
            R = b + k - 1
            if L > maxR:
                # No need to continue further
                break
            cur_sum = get_sum(L, R)
            if cur_sum > ans:
                ans = cur_sum
        
        return ans