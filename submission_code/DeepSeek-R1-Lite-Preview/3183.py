from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        for i in range(31):  # Iterate over each bit position from 0 to 30
            count = sum(1 for num in nums if num & (1 << i))
            if count >= k:
                result |= (1 << i)
        return result