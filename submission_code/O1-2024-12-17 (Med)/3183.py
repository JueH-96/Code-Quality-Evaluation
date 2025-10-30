class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # The idea is to check each bit position (from 0 to 30 inclusive)
        # and count how many of the nums have that bit set.
        # If at least k elements have bit i set, we set that bit in the result.
        
        result = 0
        for i in range(31):  # bits from 0 to 30
            mask = 1 << i
            count = sum(1 for x in nums if x & mask)
            if count >= k:
                result |= mask
        
        return result