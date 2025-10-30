from typing import List
from heapq import nlargest

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Calculate the maximum sum of subarray with length l
        def max_subarray_sum(l):
            return max(prefix_sum[i] - prefix_sum[i - l] for i in range(l, n + 1))
        
        # Calculate the strength of k subarrays with length l
        def calculate_strength(l):
            return sum((k - 2 * i) * max_subarray_sum(l) for i in range(k))
        
        # Binary search to find the maximum length of subarrays that can be chosen
        left, right = 1, n // k
        while left < right:
            mid = (left + right + 1) // 2
            if calculate_strength(mid) >= calculate_strength(mid - 1):
                left = mid
            else:
                right = mid - 1
        
        return calculate_strength(left)

# Example usage:
# sol = Solution()
# print(sol.maximumStrength([1,2,3,-1,2], 3))  # Output: 22
# print(sol.maximumStrength([12,-2,-2,-2,-2], 5))  # Output: 64
# print(sol.maximumStrength([-1,-2,-3], 1))  # Output: -1