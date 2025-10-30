from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        # Sort the array to consider subarrays in non-decreasing order.
        s = sorted(nums)
        n = len(s)
        
        # Build prefix sums for O(1) range-sum queries.
        prefixSum = [0] * (n + 1)
        for i in range(n):
            prefixSum[i+1] = prefixSum[i] + s[i]
        
        # Helper function to compute the cost to make all elements s[l..r]
        # equal to their median. We choose the lower median if the length is even,
        # which is valid for minimizing sum of absolute differences.
        def cost(l: int, r: int) -> int:
            length = r - l + 1
            m = l + (length - 1) // 2
            median_val = s[m]
            # Sum of absolute differences below or at median.
            left_cost = median_val * (m - l + 1) - (prefixSum[m+1] - prefixSum[l])
            # Sum of absolute differences above median.
            right_cost = (prefixSum[r+1] - prefixSum[m+1]) - median_val * (r - m)
            return left_cost + right_cost
        
        max_freq = 1
        
        # For each endpoint j, we use binary search to find the smallest i
        # such that the cost to unify s[i..j] fits in k.
        for j in range(n):
            lo, hi = 0, j + 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cost(mid, j) <= k:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo
            # Update the maximum frequency possible for subarray ending at j.
            if i <= j:
                max_freq = max(max_freq, j - i + 1)
        
        return max_freq