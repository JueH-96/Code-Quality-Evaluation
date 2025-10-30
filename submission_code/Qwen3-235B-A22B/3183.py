from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        for i in range(31):  # Check each bit from 0 to 30
            mask = 1 << i
            count = 0
            for num in nums:
                if num & mask:
                    count += 1
            if count >= k:
                result += mask
        return result