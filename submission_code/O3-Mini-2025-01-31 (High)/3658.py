from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        # fixed is the maximum absolute difference between two consecutive (known) integers.
        fixed = 0
        # low and high will track the minimum and maximum neighbors adjacent to any missing (-1) element.
        low = 10**9 + 1
        high = -1
        
        for i in range(n):
            if nums[i] != -1:
                # If the current known element borders a missing element, update low/high.
                if i > 0 and nums[i - 1] == -1:
                    low = min(low, nums[i])
                    high = max(high, nums[i])
                if i < n - 1 and nums[i + 1] == -1:
                    low = min(low, nums[i])
                    high = max(high, nums[i])
            # Also update fixed if two consecutive known numbers exist.
            if i > 0 and nums[i] != -1 and nums[i - 1] != -1:
                fixed = max(fixed, abs(nums[i] - nums[i - 1]))
        
        # If there is no known element adjacent to a missing element,
        # then either there are no missing values or all values are missing.
        # In either case, we can simply return fixed (which for all-missing becomes 0).
        if low == 10**9 + 1:
            return 0
        
        # It turns out that the best strategy is to replace all missing values with a single number k.
        # (The pair (x, y) can be chosen with x = y = k because they are allowed to be equal.)
        # In order to minimize the max difference at boundaries adjacent to missing values,
        # we want to minimize max(|k - low|, |k - high|). The optimal k is any value in-between and
        # the minimum possible maximum difference is ceil((high - low) / 2).
        # Using integer arithmetic, we can compute this as:
        candidate = (high - low + 1) // 2   # this is equivalent to math.ceil((high-low)/2)
        
        # The final answer must be at least as large as fixed, the max difference between adjacent knowns.
        return max(fixed, candidate)