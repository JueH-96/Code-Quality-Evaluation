class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Number of intervals
        m = len(conflictingPairs)
        # Group intervals by their left endpoint L
        L2idx = [[] for _ in range(n+2)]
        # Store right endpoints R by interval index
        R_list = [0] * m
        for idx, (a, b) in enumerate(conflictingPairs):
            L = a if a < b else b
            R = b if a < b else a
            L2idx[L].append(idx)
            R_list[idx] = R

        # unique[i] will count how many subarrays become valid only
        # by removing interval i
        unique = [0] * m
        # initial_valid = number of subarrays that do not fully cover
        # any interval (before removing anything)
        initial_valid = 0

        # We'll sweep l from n down to 1, maintaining the two smallest
        # R-values among intervals with L >= current l.
        inf = n + 1
        min1_R, min1_idx = inf, -1
        min2_R, min2_idx = inf, -1

        # Sweep l from n to 1
        for l in range(n, 0, -1):
            # Insert all intervals that start at l
            for idx in L2idx[l]:
                R = R_list[idx]
                if R < min1_R:
                    # new minimum
                    min2_R, min2_idx = min1_R, min1_idx
                    min1_R, min1_idx = R, idx
                elif R < min2_R:
                    # new second minimum
                    min2_R, min2_idx = R, idx

            # a1 = minimal R among active intervals (or n+1 if none)
            a1 = min1_R
            # Count subarrays starting at l that don't cover any interval:
            # they can end at r = l..(a1-1). So count = (a1 - l).
            # If no active intervals, a1 = n+1, so count = n+1-l.
            initial_valid += (a1 - l)

            # If there is at least one active interval, that interval
            # is min1_idx, and a2 = second minimal R (or n+1).
            if min1_idx != -1:
                a2 = min2_R
                # All subarrays [l..r] that cover the minimal-R interval
                # but cover no other interval contribute uniquely to it.
                # Count = (a2 - a1).
                unique[min1_idx] += (a2 - a1)

        # Find the best interval to remove
        best_unique = max(unique) if m > 0 else 0
        # The answer is initial_valid + best_unique
        return initial_valid + best_unique