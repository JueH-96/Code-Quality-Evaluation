from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        We will track which indices are peaks in a separate array (peak[]), and maintain a Fenwick (Binary Indexed) Tree
        to quickly update and query the count of peaks in any subarray.

        Steps:
        1. Create a function is_peak(i) to check if nums[i] is a peak.
        2. Build an initial peak[] array of 0/1 values, where peak[i] = 1 if nums[i] is a peak, else 0.
           (Note: i in [1..n-2] are the only possible peak indices.)
        3. Build a Fenwick tree over the peak[] array to efficiently query sums and update values.
        4. For each query:
           - If it is type [1, l, r], we compute the count of peaks in [l+1..r-1] using Fenwick sums.
           - If it is type [2, i, val], we update nums[i] = val, then recompute the potential peak status for
             i, i-1, i+1 (if they are valid middle indices), and update the Fenwick tree accordingly.
        5. Collect and return results for all type-1 queries.
        """

        n = len(nums)

        # Fenwick Tree (Binary Indexed Tree) implementation
        fenwicks = [0] * (n + 1)  # 1-based indexing

        def fenwicks_update(idx: int, delta: int):
            while idx <= n:
                fenwicks[idx] += delta
                idx += idx & -idx

        def fenwicks_sum(idx: int) -> int:
            s = 0
            while idx > 0:
                s += fenwicks[idx]
                idx -= idx & -idx
            return s

        def fenwicks_range_sum(left: int, right: int) -> int:
            # sum in [left..right], inclusive, 1-based
            if left > right:
                return 0
            return fenwicks_sum(right) - fenwicks_sum(left - 1)

        # Check if index i is a peak
        def is_peak(i: int) -> int:
            # i must be between 1 and n-2 to be considered a peak
            return 1 if (0 < i < n - 1 and nums[i] > nums[i - 1] and nums[i] > nums[i + 1]) else 0

        # Build the initial peak array and Fenwick tree
        peak = [0] * n
        for i in range(1, n - 1):
            peak[i] = is_peak(i)

        # Build Fenwicks
        for i in range(n):
            if peak[i] == 1:
                fenwicks_update(i + 1, 1)  # update Fenwicks at 1-based index i+1

        answers = []

        for q in queries:
            if q[0] == 1:
                # query of type [1, l, r]
                l, r = q[1], q[2]
                # The subarray is nums[l..r], we only count peaks in the range (l+1..r-1)
                # if r - l < 2, no space for a peak
                if r - l < 2:
                    answers.append(0)
                else:
                    # query fenwicks in [l+2..r], 1-based => indices are [l+1..r-1] zero-based
                    left_idx = l + 2  # 1-based => l+1+1
                    right_idx = r  # 1-based => r
                    ans = fenwicks_range_sum(left_idx, right_idx)
                    answers.append(ans)

            else:
                # query of type [2, index, val]
                i, val = q[1], q[2]
                # update nums[i] = val
                nums[i] = val

                # check i, i-1, i+1 in range [1..n-2] if they are peaks
                # each time we recalc is_peak and update Fenwicks if a change
                for j in [i - 1, i, i + 1]:
                    if 1 <= j <= n - 2:
                        new_val = is_peak(j)
                        old_val = peak[j]
                        if new_val != old_val:
                            peak[j] = new_val
                            fenwicks_update(j + 1, new_val - old_val)  # 1-based index in Fenwicks

        return answers