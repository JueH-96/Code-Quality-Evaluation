from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        max_bit = max_num.bit_length()
        k_or = 0
        
        for bit in range(max_bit):
            count = sum(1 for num in nums if num & (1 << bit))
            if count >= k:
                k_or |= (1 << bit)
        
        return k_or