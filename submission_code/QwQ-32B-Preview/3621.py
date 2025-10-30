from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if there are elements less than k that are not equal to k
        if any(num < k for num in nums if num != k):
            return -1
        
        # Find unique values greater than k
        unique_values = set(nums)
        values_greater_than_k = [num for num in unique_values if num > k]
        
        # The number of operations is the number of unique values greater than k
        return len(values_greater_than_k)