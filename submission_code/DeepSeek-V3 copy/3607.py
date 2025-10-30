from typing import List
import math

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        # Precompute the greatest proper divisor for each possible number up to 10^6
        max_num = max(nums)
        gpd = [1] * (max_num + 1)
        for i in range(2, max_num + 1):
            for j in range(i * 2, max_num + 1, i):
                gpd[j] = i
        
        # Iterate from the end to the start
        operations = 0
        for i in range(n-2, -1, -1):
            if nums[i] > nums[i+1]:
                # We need to reduce nums[i] to be <= nums[i+1]
                # Find the smallest possible value for nums[i] that is <= nums[i+1]
                # and can be obtained by dividing nums[i] by its greatest proper divisor
                current = nums[i]
                while current > nums[i+1]:
                    if gpd[current] == 1:
                        # Cannot reduce further
                        return -1
                    current = current // gpd[current]
                    operations += 1
                nums[i] = current
        
        return operations