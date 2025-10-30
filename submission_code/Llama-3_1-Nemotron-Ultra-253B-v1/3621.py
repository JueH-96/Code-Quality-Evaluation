from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        if any(x < k for x in nums):
            return -1
        
        # Check if all elements are already k
        if all(x == k for x in nums):
            return 0
        
        # Collect unique values greater than k and sort them in decreasing order
        unique = sorted(list({x for x in nums if x > k}), reverse=True)
        
        # The number of operations is the number of unique values
        return len(unique)