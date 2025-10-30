from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        max_bits = max(nums).bit_length()
        
        for i in range(max_bits):
            count = 0
            for num in nums:
                if (num >> i) & 1:
                    count += 1
            if count >= k:
                result |= (1 << i)
        
        return result