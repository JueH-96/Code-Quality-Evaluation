from typing import List

class Solution:
    def partitionDisjoint(self, nums: List[int]) -> int:
        n = len(nums)
        left_max = [0] * n
        right_min = [0] * n
        
        left_max[0] = nums[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], nums[i])
        
        right_min[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            right_min[i] = min(right_min[i+1], nums[i])
        
        for i in range(n-1):
            if left_max[i] <= right_min[i+1]:
                return i + 1

# The above solution is for the problem "Partition Array" which is similar to the given problem.
# However, the given problem is asking for the minimum value of the partition, not the partition index.
# We can modify the solution to find the minimum value of the partition.

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        
        min_diff = float('inf')
        for i in range(1, n):
            diff = abs(nums[i-1] - nums[i])
            min_diff = min(min_diff, diff)
        
        return min_diff