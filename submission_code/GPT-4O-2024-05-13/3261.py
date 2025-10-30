from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # If k is greater than or equal to the length of nums - 1, we can reduce the array to a single element
        if k >= n - 1:
            result = nums[0]
            for num in nums:
                result &= num
            return result
        
        # Otherwise, we need to find the minimum possible OR after at most k operations
        min_or = float('inf')
        
        # Iterate over all possible subarrays of length k+1
        for i in range(n - k):
            current_and = nums[i]
            for j in range(i + 1, i + k + 1):
                current_and &= nums[j]
            min_or = min(min_or, current_and)
        
        return min_or