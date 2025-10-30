from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if all elements are >= k
        if any(num < k for num in nums):
            return -1
        
        # Collect unique values greater than k
        unique_values = set(num for num in nums if num > k)
        
        # The number of operations is the number of unique values greater than k
        return len(unique_values)