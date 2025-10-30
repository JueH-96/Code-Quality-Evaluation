class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        current_and = (1 << 20) - 1  # Initialize with all bits set
        for num in nums:
            current_and &= num
        if current_and != 0:
            return 1
        count = 0
        current_and = (1 << 20) - 1
        for num in nums:
            current_and &= num
            if current_and == 0:
                count += 1
                current_and = (1 << 20) - 1
        return count