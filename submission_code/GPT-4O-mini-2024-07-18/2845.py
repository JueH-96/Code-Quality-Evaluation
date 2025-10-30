class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_partition_value = float('inf')
        
        for i in range(len(nums) - 1):
            partition_value = abs(nums[i] - nums[i + 1])
            min_partition_value = min(min_partition_value, partition_value)
        
        return min_partition_value