class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # The result integer we'll build by setting its bits as required
        result = 0
        
        # We'll check bit positions from 0 through 30 (since 2^31 is given as upper bound)
        for bit in range(31):
            mask = 1 << bit
            # Count how many elements in nums have the current bit set
            count = sum(1 for num in nums if num & mask)
            
            # If at least k elements have this bit set, set it in the result
            if count >= k:
                result |= mask
        
        return result