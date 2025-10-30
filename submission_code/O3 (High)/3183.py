from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # There are at most 31 relevant bits since nums[i] < 2^31
        BIT_LIMIT = 31
        
        bit_count = [0] * BIT_LIMIT
        
        # Count how many numbers have each bit set
        for num in nums:
            for bit in range(BIT_LIMIT):
                if num >> bit & 1:
                    bit_count[bit] += 1
        
        # Build the resulting K-or
        answer = 0
        for bit in range(BIT_LIMIT):
            if bit_count[bit] >= k:
                answer |= 1 << bit
        
        return answer