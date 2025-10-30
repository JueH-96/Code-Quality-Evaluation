class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        from functools import reduce
        import operator

        S = reduce(operator.and_, nums)
        count = 0
        current = ~0  # Initialize with all bits set

        for num in nums:
            current &= num
            if current == S:
                count += 1
                current = ~0  # Reset for the next subarray

        return count