from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        max_bit_length = max(num.bit_length() for num in nums)
        k_or_result = 0

        for i in range(max_bit_length):
            bit_count = sum((num >> i) & 1 for num in nums)
            if bit_count >= k:
                k_or_result |= (1 << i)

        return k_or_result