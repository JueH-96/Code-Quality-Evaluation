from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        result = 0
        for b in range(30, -1, -1):
            mask = 1 << b
            current_and = None
            count = 0
            for num in nums:
                if current_and is None:
                    current_and = num
                else:
                    current_and &= num
                if (current_and & mask) == 0:
                    count += 1
                    current_and = None
            if count < (n - k):
                result |= mask
        return result