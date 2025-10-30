from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        # Sort the array.
        nums.sort()
        n = len(nums)
        # Build prefix sum array for constant time range sum queries.
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Helper function to compute the cost to convert all numbers
        # in the window [l, r] (inclusive) into a common value chosen
        # optimally (the median of the window).
        def cost(l: int, r: int) -> int:
            L = r - l + 1
            # median index (if even, we choose the first element of the right half,
            # which works because the sum of absolute differences is constant in the median interval)
            m = l + L // 2
            # cost is sum_{i=l}^{m-1} (nums[m] - nums[i]) + sum_{i=m+1}^{r} (nums[i] - nums[m])
            left_cost = nums[m] * (m - l) - (prefix[m] - prefix[l])
            right_cost = (prefix[r+1] - prefix[m+1]) - nums[m] * (r - m)
            return left_cost + right_cost
        
        res = 1
        l = 0
        # Use two pointers (sliding window) over the sorted array.
        for r in range(n):
            # while the cost to convert window [l, r] into a single value exceeds k, 
            # shrink the window from the left.
            while l <= r and cost(l, r) > k:
                l += 1
            # update the result with the current window size.
            res = max(res, r - l + 1)
            
        return res