class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # The K-or of nums is a value whose bit i is set if and only if
        # there are at least k elements of nums having bit i set.

        # Initialize the result to 0
        k_or_value = 0

        # Since nums[i] can be up to 2^31 - 1, we'll check bits from 0 to 31
        for bit in range(32):
            count = 0
            mask = 1 << bit
            # Count how many elements have this bit set
            for num in nums:
                if num & mask:
                    count += 1
            # If at least k elements have this bit set, add it to k_or_value
            if count >= k:
                k_or_value |= mask
        
        return k_or_value