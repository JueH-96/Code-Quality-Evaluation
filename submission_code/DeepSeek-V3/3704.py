class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        if total_sum % 2 == 0:
            return len(nums) - 1
        else:
            return 0