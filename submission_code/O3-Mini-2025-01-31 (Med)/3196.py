from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        # Sort the array. In an optimal solution, the best group of numbers to convert
        # into one value are consecutively arranged when sorted.
        nums.sort()
        n = len(nums)
        
        # Build a prefix sum array for fast sum queries.
        prefix = [0] * n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i - 1] + nums[i]
        
        # Helper function: Given a contiguous window [l, r] of length L = r-l+1,
        # we want to compute the cost to convert all numbers in that window
        # into one common value. With both increases and decreases allowed,
        # the optimal target value is the median of the window.
        # For window [l, r]: let m = l + (L - 1) // 2 (select one median in case of even length).
        # Then the cost is:
        #   cost = (nums[m]*(# left numbers) - sum of left numbers) +
        #          (sum of right numbers - nums[m]*(# right numbers)).
        def window_cost(l: int, r: int) -> int:
            L = r - l + 1
            mid = l + (L - 1) // 2  # median index
            left_count = mid - l
            right_count = r - mid
            left_sum = prefix[mid - 1] - (prefix[l - 1] if l > 0 else 0) if left_count > 0 else 0
            right_sum = prefix[r] - prefix[mid] if right_count > 0 else 0
            cost = nums[mid] * left_count - left_sum + right_sum - nums[mid] * right_count
            return cost
        
        # We want the maximum frequency we can achieve.
        # Notice that if we can change a group of L numbers to one value, that L is
        # a candidate frequency. In a sorted array, the best L numbers to convert
        # are contiguous. So our task reduces to:
        #   "Find the maximum L (window length) such that, for some contiguous window,
        #    the cost to convert all its elements to the median is <= k."
        #
        # For each candidate window length L, we can slide over the sorted array and
        # check if any window is feasible.
        def can_make(L: int) -> bool:
            # Check all windows of length L
            for start in range(0, n - L + 1):
                end = start + L - 1
                if window_cost(start, end) <= k:
                    return True
            return False
        
        # Binary search for the answer between 1 and n.
        lo, hi = 1, n
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_make(mid):
                best = mid
                lo = mid + 1  # try for a bigger window
            else:
                hi = mid - 1  # try a smaller window
        return best