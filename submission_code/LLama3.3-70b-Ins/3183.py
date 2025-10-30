from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # Initialize the result as 0
        result = 0
        
        # Iterate over each bit position (up to 31 bits)
        for i in range(31):
            # Count the number of elements in nums where the current bit is set
            count = sum(1 for num in nums if (num >> i) & 1)
            
            # If the count is at least k, set the current bit in the result
            if count >= k:
                result |= 1 << i
        
        return result