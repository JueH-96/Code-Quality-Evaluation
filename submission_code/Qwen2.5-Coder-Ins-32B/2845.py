from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_partition_value = float('inf')
        for i in range(1, len(nums)):
            min_partition_value = min(min_partition_value, abs(nums[i] - nums[i - 1]))
        return min_partition_value