from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        """
        After at most k ±1 operations we want as many equal numbers as possible.
        For a group of elements the cheapest common value is their median, and
        its cost is the sum of distances to that median.

        In a sorted array the optimal group is always a contiguous block, so for
        every block we are able to compute that minimal cost in O(1) by using
        prefix sums.  We binary–search on the answer (group size) and test each
        candidate size in O(n), resulting in an O(n log n) solution.
        """
        a = sorted(nums)
        n = len(a)

        # prefix[i] = sum of first i elements, prefix[0] = 0
        pref = [0] * (n + 1)
        for i, v in enumerate(a):
            pref[i + 1] = pref[i] + v

        # test whether some block of length `length` can be equalised within k
        def can(length: int) -> bool:
            for left in range(0, n - length + 1):
                right = left + length - 1
                mid = left + length // 2           # index of upper median
                median = a[mid]

                # cost for the left half  (elements < median)
                left_sum = pref[mid] - pref[left]          # sum a[left .. mid-1]
                cost_left = median * (mid - left) - left_sum

                # cost for the right half (elements > median)
                right_sum = pref[right + 1] - pref[mid + 1]  # sum a[mid+1 .. right]
                cost_right = right_sum - median * (right - mid)

                if cost_left + cost_right <= k:
                    return True
            return False

        # binary search for the maximum feasible length
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo