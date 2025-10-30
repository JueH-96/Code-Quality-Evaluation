class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        if total_sum % 2 == 0:  # If sum is even, all partitions have even difference
            return len(nums) - 1
        else:  # If sum is odd, no partition has even difference
            return 0