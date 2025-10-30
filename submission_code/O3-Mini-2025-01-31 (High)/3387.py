from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # First, sort the array.
        nums.sort()
        n = len(nums)
        # For both odd and even lengths, by problem statement, the median
        # is defined as the element at index n//2 (0-indexed) when sorted.
        m = n // 2
        ops = 0
        
        # For the median to be k after modifications, in the final sorted array:
        # • Every element at index i >= m must be at least k; otherwise, if an element
        #   is too small, then when sorted it could become the median or affect the order.
        # • Every element at index i <= m must be at most k.
        #
        # Thus, we can process the two halves separately:
        # 1. For indices from m to end, if an element is below k,
        #    we need to increase it (cost = k - value).
        for i in range(m, n):
            if nums[i] < k:
                ops += k - nums[i]
                
        # 2. For indices from 0 to m, if an element is above k,
        #    we need to decrease it (cost = value - k).
        for i in range(0, m + 1):
            if nums[i] > k:
                ops += nums[i] - k
        
        return ops

# Example usage:
# sol = Solution()
# print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 4))  # Output: 2
# print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 7))  # Output: 3
# print(sol.minOperationsToMakeMedianK([1,2,3,4,5,6], 4))  # Output: 0