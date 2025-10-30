class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        """
        We want to see if there exist two parallel cuts (either horizontal or vertical)
        that partition the grid into exactly three non-empty sections, each containing
        at least one rectangle, and no rectangle is intersected by a cut.

        Key points/observations:
          1. The cuts must be strictly between 0 and n (i.e., no cut on the boundary).
          2. A cut line cannot pass through the interior of any rectangle. In other words,
             if a rectangle spans from sy to ey (for horizontal cuts), then a horizontal
             cut y_cut must not satisfy sy < y_cut < ey. The same logic applies for x-cuts.
          3. Because the rectangles do not overlap, we only need to ensure each rectangle
             is wholly in one of the three sections formed by the two parallel cuts.
          4. Each of the three sections formed by the cuts must contain at least one rectangle.

        Approach (high-level):
          • We attempt to find two valid horizontal cuts. If successful, return True.
            Otherwise, try two valid vertical cuts.
            If neither horizontal nor vertical cuts can partition the rectangles properly,
            return False.
          
          • To find valid cuts (taking horizontal as example):
             a) Collect all unique y-edges from the rectangles (both bottom, sy, and top, ey).
             b) Sort these edges to get a sorted array Y. Between any consecutive pair (Y[i], Y[i+1]),
                we have a potential open interval for a horizontal cut. But first we must check
                if that entire open interval (Y[i], Y[i+1]) is "blocked" by the interior of 
                any rectangle (i.e., if there's any rectangle that covers part of that open interval).
                We do so by a sweep-line or counting approach.
             c) The midpoints (or anything strictly between (Y[i], Y[i+1])) of unblocked intervals
                are our valid cut lines.
             d) Precompute for any candidate cut c:
                  countBelow(c) = number of rectangles fully below c
                  countAbove(c) = number of rectangles fully above c
                So that for two cuts c1 < c2, 
                  • bottom section count = countBelow(c1)
                  • top section count    = countAbove(c2)
                  • middle section count = totalRectangles - (bottom + top)
             e) We need bottom >= 1, middle >= 1, top >= 1, and c1, c2 are both valid (unblocked).
             f) We check if there exist c1, c2 (with c1 < c2) that satisfy these counts. We can do
                this in O(M) or O(M log M) time after constructing the candidate cut array,
                where M is the number of candidate cut positions.

          • We do exactly the same steps for vertical cuts (just use x-coordinates).

        Complexity:
          - Let m = number of rectangles (up to 1e5).
          - We gather up to 2m unique edges for either horizontal or vertical sides.
          - Sorting edges is O(m log m). Building prefix/suffix arrays (countBelow / countAbove)
            can be done with binary search in O(m log m). Then a two-pointer or binary search
            approach to find suitable pairs c1 < c2 is O(m).
          - Overall O(m log m) should be feasible for m up to 1e5.

        Let's implement the solution following this outline.
        """

        import bisect

        # 1) Build a function that, given an array of rectangles and an index of interest (0 or 1),
        #    returns whether two valid cuts exist in that dimension.
        #    If index=0, that means we check vertical dimension using [x1, x2].
        #    If index=1, that means we check horizontal dimension using [y1, y2].
        def canMakeTwoCuts(rects, dimIndex=1):
            # dimIndex = 0 -> x-based cuts
            # dimIndex = 1 -> y-based cuts

            # Collect all rectangle edges in the chosen dimension.
            starts = []  # sy or sx
            ends   = []  # ey or ex
            for (sx, sy, ex, ey) in rects:
                if dimIndex == 1:
                    starts.append(sy)
                    ends.append(ey)
                else:
                    starts.append(sx)
                    ends.append(ex)

            # Sort all unique edges
            all_edges = sorted(set(starts + ends))
            total_rects = len(rects)

            # We'll do a line-sweep to find which intervals (all_edges[i], all_edges[i+1])
            # are valid/unblocked. That is, no rectangle intersects that open interval.
            # Prepare events: (+1 when we enter a rectangle, -1 when we exit).
            # For the dimension in question, we treat each rectangle as an interval (start, end).
            # We'll increment a counter at 'start' and decrement at 'end'.
            events = []
            for st, ed in zip(starts, ends):
                events.append((st, +1))
                events.append((ed, -1))
            events.sort()

            # We'll sweep from min edge to max edge, tracking how many rectangles cover
            # the current "y" (or "x") range. Then if coverage>0 over an interval,
            # that interval is blocked. If coverage=0, it is unblocked.
            coverage = 0
            idxE = 0  # index to events
            nE = len(events)
            L = len(all_edges)

            blocked = [False]*(L-1)  # blocked[i] -> if (all_edges[i], all_edges[i+1]) is blocked

            for i in range(L-1):
                # Move the sweep line from all_edges[i] to all_edges[i+1] (but open interval).
                # First, update coverage at all events <= all_edges[i].
                while idxE < nE and events[idxE][0] <= all_edges[i]:
                    coverage += events[idxE][1]
                    idxE += 1

                # The interval (all_edges[i], all_edges[i+1]) is blocked if coverage>0.
                blocked[i] = (coverage > 0)

                # Then, before we move to the next interval, also update coverage
                # at events that are <= all_edges[i+1] if needed,
                # but we only do that at the start of the next iteration or if we want
                # a stricter approach. We'll handle the increments/decrements as
                # we proceed from left to right fully.
                # Actually, we can partially update coverage so that
                # we remain consistent for the next iteration:
                while idxE < nE and events[idxE][0] <= all_edges[i+1]:
                    coverage += events[idxE][1]
                    idxE += 1

            # validCuts is the list of midpoints for each unblocked interval
            validCuts = []
            for i in range(L-1):
                if not blocked[i]:
                    # pick a representative point in (all_edges[i], all_edges[i+1]), for instance, midpoint
                    mid = (all_edges[i] + all_edges[i+1]) / 2
                    validCuts.append(mid)

            if len(validCuts) < 2:
                return False  # we need at least 2 valid cuts to form 3 sections

            # Precompute sorted starts and ends to quickly compute:
            #   countBelow(c) = #rects with end <= c
            #   countAbove(c) = #rects with start >= c
            starts_sorted = sorted(starts)
            ends_sorted   = sorted(ends)

            def countBelow(c):
                # number of rectangles whose 'end' <= c
                # we can bisect_right in ends_sorted
                return bisect.bisect_right(ends_sorted, c)

            def countAbove(c):
                # number of rectangles whose 'start' >= c
                # this is total_rects - index of first rectangle start < c
                # or using bisect_left in starts_sorted
                pos = bisect.bisect_left(starts_sorted, c)
                return total_rects - pos

            # Now we want to see if there is i < j such that:
            #   bottom = countBelow(validCuts[i]) >= 1
            #   top = countAbove(validCuts[j]) >= 1
            #   middle = total_rects - (bottom + top) >= 1
            #   => bottom + top <= total_rects - 1
            # We'll do a two-pointer approach from left to right and right to left.

            # Let's store B[i] = countBelow(validCuts[i]) for each i
            # and A[i] = countAbove(validCuts[i]) for each i
            B = [countBelow(cut) for cut in validCuts]
            A = [countAbove(cut) for cut in validCuts]

            # We'll move a pointer i from left to right to pick c1,
            # and a pointer j from right to left to pick c2.
            # We only need to ensure i < j (i.e. validCuts[i] < validCuts[j]) and
            # check if B[i]+A[j] <= total_rects-1 and also B[i] >=1, A[j] >=1.

            # Because validCuts is strictly increasing, i<j => validCuts[i] < validCuts[j].
            # We'll do a standard approach to see if there's at least one pair meeting conditions.
            left = 0
            right = len(validCuts) - 1

            # We can do a while left < len(validCuts) and right >= 0, ensuring left < right
            while left < len(validCuts) and right >= 0 and left < right:
                if B[left] >= 1 and A[right] >= 1:
                    if B[left] + A[right] <= total_rects - 1:
                        # We found a valid partition
                        return True
                    # If B[left] + A[right] is large, we want it smaller => move right--
                    # If B[left] + A[right] is small, we want it bigger => move left++,
                    # but we just need existence, so let's systematically move.
                    # We want to try to reduce the sum B[left]+A[right], so we do right -= 1
                    # But that might lose the c2>c1 property if right <= left. We'll keep that in check.
                    if B[left] + A[right] > total_rects - 1:
                        right -= 1
                    else:
                        left += 1
                else:
                    # If B[left]<1, we should move left up to increase B[left]
                    if B[left] < 1:
                        left += 1
                    # If A[right]<1, we move right down to possibly increase A[right]
                    # but moving right down typically might not help if starts are sorted ascending
                    # Actually, if A[right]<1, we might need to move right further down 
                    # to get a cut that is even smaller, but that won't help because
                    # the start threshold is c. If c is lower, that means
                    # fewer rectangles have start >= c => actually A[..] might get bigger
                    # the smaller c is. So let's do right -= 1 in that scenario.
                    elif A[right] < 1:
                        right -= 1

            return False

        # Try horizontal cuts (dimIndex=1)
        if canMakeTwoCuts(rectangles, dimIndex=1):
            return True

        # Try vertical cuts (dimIndex=0)
        if canMakeTwoCuts(rectangles, dimIndex=0):
            return True

        # If neither worked, return False
        return False