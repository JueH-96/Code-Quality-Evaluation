class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        for bit in range(31):  # Check bits from 0 to 30
            mask = 1 << bit
            count = sum(1 for num in nums if num & mask != 0)
            if count >= k:
                result |= mask
        return result