class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        return len(nums) - 1 if total_sum % 2 == 0 else 0