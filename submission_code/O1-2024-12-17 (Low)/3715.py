class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        """
        We have disjoint segments [l_i, r_i] each giving c_i coins per bag in that range.
        We want to pick k consecutive integer positions on the number line to maximize total coins.

        Key idea:
        - Because the segments are disjoint and can be very large (up to 1e9 in coordinate),
          we cannot iterate naively over all positions.
        - Observe that the coin function f(x) (coins at position x) is piecewise constant,
          changing only at segment boundaries.
        - A crucial insight is that if we slide a length-k window over this piecewise constant
          function, the sum only changes when we cross a boundary. Hence it suffices to
          consider windows whose left edge is exactly at a boundary.

        Steps:
          1) Convert each segment [l, r, c] into two "events":
             (l, +c) and (r+1, -c). Sort by the coordinate.
          2) Sweep through these events to build a list of "parts":
             parts = [(start, end, val)] with end exclusive, val constant in [start, end).
             In effect, this constructs the piecewise constant function across the union
             of all segments (disjoint).
          3) Build prefix sums over these parts:
             prefixLen[i] = total length from parts[0]..parts[i-1].
             prefixCoins[i] = total coins from parts[0]..parts[i-1].
             so prefixCoins[i] = prefixCoins[i-1] + (length_of_part_{i-1} * val_{i-1}).
          4) For each part i, consider a window that starts exactly at parts[i].start
             and ends at parts[i].start + k. That is [startCoord, endCoord).
             Use binary search on the part 'end' values to find how many parts
             are fully covered by [startCoord, endCoord). Then add partial coverage
             of the next part if needed.
          5) Keep track of the maximum sum encountered.

        This runs in O(N log N) time for N = len(coins), which is feasible for up to 1e5 segments.
        """

        import bisect

        # Step 1: Build the events array
        events = []
        for l, r, c in coins:
            # start event
            events.append((l, c))
            # end event: r+1 gets -c
            events.append((r+1, -c))
        # Sort by coordinate
        events.sort()

        # Step 2: Build the parts array (start, end, val), end is exclusive.
        parts = []
        cur_val = 0  # current coin value in the region
        prev_coord = events[0][0]  # where the current region starts

        for i in range(len(events)):
            coord, delta = events[i]
            if coord > prev_coord:
                # we have a region [prev_coord, coord) with coin = cur_val
                if cur_val != 0:
                    parts.append((prev_coord, coord, cur_val))
                prev_coord = coord
            cur_val += delta

        # If there is no end event that extends further, we do not add anything
        # beyond the last event since coin value becomes zero or the events have ended.

        # Edge case: if no valid parts at all, answer is 0
        if not parts:
            return 0 if k > 0 else 0

        # Step 3: Build prefix sums
        # prefixLen[i] = total length from parts[0] to parts[i-1]
        # prefixCoins[i] = total coins from parts[0] to parts[i-1]
        n = len(parts)
        prefixLen = [0]*(n+1)
        prefixCoins = [0]*(n+1)

        for i in range(n):
            (s, e, val) = parts[i]
            length = e - s
            prefixLen[i+1] = prefixLen[i] + length
            prefixCoins[i+1] = prefixCoins[i] + length * val

        # For binary searching by end-coordinates:
        # We'll keep an array of the exclusive ends: partEnds[i] = parts[i].end
        partEnds = [parts[i][1] for i in range(n)]

        ans = 0

        # Helper function to get how many coins in [startCoord, endCoord).
        # We'll do so by:
        #   1) Find the largest j where parts[j].end <= endCoord => j = right_index - 1
        #   2) Sum full coverage from i..j if j >= i
        #   3) Add partial coverage from j+1 if it intersects
        # But we are enumerating i from 0..n-1, so the sub-interval starts at parts[i].start.
        # If the entire sub-interval fits inside the same part i, handle that as a special case.
        # Otherwise do the approach above.
        for i in range(n):
            startCoord = parts[i][0]
            val_i = parts[i][2]
            endCoord = startCoord + k

            # If the sub-interval is fully contained within part i:
            if endCoord <= parts[i][1]:
                # entire length k is inside this one part
                total = k * val_i
                ans = max(ans, total)
                continue

            # Otherwise, find how many parts are fully covered
            # We want the insertion point of endCoord in partEnds
            # j = bisect_left(partEnds, endCoord) gives the first index with partEnds[j] >= endCoord
            # so j-1 is the last index with partEnds[j-1] < endCoord
            # but we want the ones strictly <= endCoord, so let's use j = bisect.bisect_left(...)
            j = bisect.bisect_left(partEnds, endCoord) - 1

            if j < i:
                # Means endCoord is before parts[i].end,
                # but we handled the case endCoord <= parts[i].end above.
                # This would occur if endCoord is <= parts[i].start, which can't happen
                # since endCoord = parts[i].start + k > parts[i].start. So ignore.
                pass

            # sumFull is the fully covered parts i..j if j >= i
            sumFull = 0
            if j >= i:
                sumFull = prefixCoins[j+1] - prefixCoins[i]

            # partial coverage from j+1 if the sub-interval intersects that region
            partial = 0
            next_idx = j+1
            if 0 <= next_idx < n:
                s2, e2, val2 = parts[next_idx]
                # If [s2, e2) intersects with [startCoord, endCoord),
                # intersection is [s2, min(e2, endCoord))
                if s2 < endCoord:
                    inter_len = min(e2, endCoord) - s2
                    if inter_len > 0:
                        partial = inter_len * val2

            total = sumFull + partial
            ans = max(ans, total)

        return ans