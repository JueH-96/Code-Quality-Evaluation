class Solution:
    def findKOr(self, nums: list[int], k: int) -> int:
        result = 0
        # iterate over bit positions 0 to 30 (since 2^31 > any number in nums)
        for bit in range(31):
            mask = 1 << bit
            count = sum(1 for num in nums if num & mask)
            # if the bit appears in at least k numbers, set this bit in result
            if count >= k:
                result |= mask
        return result