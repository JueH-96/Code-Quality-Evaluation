from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for x in nums:
            current_xor ^= x
        if current_xor == k:
            return 0
        
        original_nums = list(nums)
        for i in range(len(nums)):
            for j in range(21): # Check bit positions up to 20 (for numbers up to 10^6)
                nums[i] ^= (1 << j)
                new_xor = 0
                for x in nums:
                    new_xor ^= x
                if new_xor == k:
                    return 1
                nums[i] = original_nums[i] # Backtrack, restore original value for this index

        return 2