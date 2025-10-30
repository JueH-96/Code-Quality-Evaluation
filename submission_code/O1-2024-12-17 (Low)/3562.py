class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        """
        We want to pick up to 4 non-overlapping intervals with maximum total weight,
        returning the lexicographically smallest list of their (1-based) indices among
        all optimal choices.

        Approach:
          1) Sort the intervals by their ending point r_i in ascending order,
             keeping track of the original index.
          2) For each interval i (in sorted order), find p[i], the largest j < i
             such that intervals[j].r < intervals[i].l (strictly less, because
             shared boundary => overlapping).
          3) Use dynamic programming with up to k = 4 intervals:
             Let dp[k][i] = (best sum using up to k intervals among the first i sorted
                            intervals, list of chosen original indices in ascending order).
             We only need dp1, dp2, dp3, dp4 (for k=1..4). Each dp array has size n+1
             (dp[k][0] = (0, [])).
             
             Recurrence for i from 1..n:
               - dp1[i] = best of dp1[i-1] and (weight[i], [orig_i])
               - dp2[i] = best of dp2[i-1] and (dp1[p[i]].sum + weight[i], dp1[p[i]].list + [orig_i])
               - dp3[i] = best of dp3[i-1] and (dp2[p[i]].sum + weight[i], dp2[p[i]].list + [orig_i])
               - dp4[i] = best of dp4[i-1] and (dp3[p[i]].sum + weight[i], dp3[p[i]].list + [orig_i])
             where "best" means:
               (a) Larger sum is better.
               (b) If sums are equal, compare chosen index lists lexicographically.
             
          4) At the end, pick the best among dp1[n], dp2[n], dp3[n], dp4[n]
             according to the same "best" definition. That gives the maximum total
             weight and the lexicographically smallest index set if there's a tie.
          5) Return that final list (converted from 0-based to 1-based).
        """

        import bisect

        # 1) Attach original indices and sort by r in ascending order
        #    We'll store intervals as (l, r, w, orig_index).
        arr = [(intervals[i][0], intervals[i][1], intervals[i][2], i) 
               for i in range(len(intervals))]
        arr.sort(key=lambda x: x[1])  # sort by r ascending

        n = len(arr)
        # Precompute the r-values in sorted order, for binary searching p[i].
        rvals = [arr[i][1] for i in range(n)]
        lvals = [arr[i][0] for i in range(n)]

        # 2) Compute p[i]: the rightmost index j < i with arr[j].r < arr[i].l
        #    We can do this by binary searching in rvals for arr[i].l.
        p = [0]*n
        for i in range(n):
            l_i = arr[i][0]
            # find largest j < i with rvals[j] < l_i
            # we do bisect_left(rvals, l_i) which gives the position of l_i in rvals
            # we want one less than that position
            pos = bisect.bisect_left(rvals, l_i)
            p[i] = pos - 1  # j = pos-1 might be -1 if none is valid

        # We'll store dp1, dp2, dp3, dp4 each as lists of n+1 elements.
        # Each dpX[i] = (best_sum, best_list_of_indices)
        # Use a helper to do "better" comparisons.
        def better(a, b):
            # returns True if a is strictly better or equal to b by our rules
            # but we only want to use it to decide if we replace b with a,
            # so we want True if a > b in the ordering.
            sumA, listA = a
            sumB, listB = b
            if sumA > sumB:
                return True
            if sumA < sumB:
                return False
            # sums equal => lex-compare on the lists
            # standard python list comparison is lexicographic
            # but also if they match on all positions up to min length, 
            # the shorter list is lexicographically smaller
            # which is exactly how Python compares lists.
            return listA < listB

        # Initialize dp arrays
        dp1 = [(0, [])]*(n+1)  # dp1[i] = best sum using up to 1 interval among first i
        dp2 = [(0, [])]*(n+1)
        dp3 = [(0, [])]*(n+1)
        dp4 = [(0, [])]*(n+1)

        # We'll do 1-based indexing in the dp arrays, but arr/p are 0-based.
        # So when we refer to the i-th interval in dp, we actually refer to arr[i-1].
        # We'll need to be careful with indexing.

        # "copy structure" trick: if dpX[i-1] is stored as a tuple, we must ensure
        # not to inadvertently mutate it. We'll rebuild new tuples as needed.

        for i in range(1, n+1):
            (l_i, r_i, w_i, orig_i) = arr[i-1]  # the i-th interval in 1-based sense
            # Update dp1[i]:
            # candidate1 = dp1[i-1]
            candidate1 = dp1[i-1]
            # candidate2 = use just this interval alone => (w_i, [orig_i])
            candidate2 = (w_i, [orig_i])
            if better(candidate1, candidate2):
                dp1[i] = candidate1
            else:
                dp1[i] = candidate2

            # dp2[i]:
            # candidate1 = dp2[i-1]
            candidate1 = dp2[i-1]
            # candidate2 = dp1[p[i-1]] + (w_i, [orig_i]) if p[i-1] >= 0
            candidate2 = candidate1  # default to something
            idx_p = p[i-1]  # p array is 0-based
            if idx_p >= 0:
                sumPrev, listPrev = dp1[idx_p+1]  # dp1 is 1-based
                newList = listPrev + [orig_i]
                newList = sorted(newList)  # ensure ascending by original index
                candidate2 = (sumPrev + w_i, newList)
            else:
                # means no non-overlapping predecessor => can only combine with dp1[0] = (0, [])
                candidate2 = (dp1[0][0] + w_i, [orig_i])

            if better(candidate1, candidate2):
                dp2[i] = candidate1
            else:
                dp2[i] = candidate2

            # dp3[i]:
            # candidate1 = dp3[i-1]
            candidate1 = dp3[i-1]
            candidate2 = candidate1  # default
            if idx_p >= 0:
                sumPrev, listPrev = dp2[idx_p+1] 
                newList = listPrev + [orig_i]
                newList = sorted(newList)
                candidate2 = (sumPrev + w_i, newList)
            else:
                # combine with dp2[0] = (0, []) if that makes sense
                candidate2 = (dp2[0][0] + w_i, [orig_i])

            if better(candidate1, candidate2):
                dp3[i] = candidate1
            else:
                dp3[i] = candidate2

            # dp4[i]:
            # candidate1 = dp4[i-1]
            candidate1 = dp4[i-1]
            candidate2 = candidate1  # default
            if idx_p >= 0:
                sumPrev, listPrev = dp3[idx_p+1]
                newList = listPrev + [orig_i]
                newList = sorted(newList)
                candidate2 = (sumPrev + w_i, newList)
            else:
                candidate2 = (dp3[0][0] + w_i, [orig_i])

            if better(candidate1, candidate2):
                dp4[i] = candidate1
            else:
                dp4[i] = candidate2

        # Now pick the best among dp1[n], dp2[n], dp3[n], dp4[n],
        # according to the "better" function, but also preferring fewer intervals
        # if sums tie (the "better" function already accounts for that).
        candidates = [dp1[n], dp2[n], dp3[n], dp4[n]]
        best = candidates[0]
        for c in candidates[1:]:
            if better(c, best):
                best = c

        # best is (sum, list_of_original_indices), all 0-based
        chosen = best[1]
        # Convert to 1-based and sort (should still be sorted, but just ensure).
        chosen = sorted(chosen)
        answer = [x+1 for x in chosen]

        return answer