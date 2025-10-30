from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Compute the required adjustment (diff) for each index.
        # diff[i] = target[i] - nums[i]
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        
        # The idea:
        # We can adjust any subarray by 1 in one operation.
        # It's optimal to cover as many indices as possible that need the same increment or decrement.
        # Thus, the minimal operations equals:
        #   abs(diff[0]) + sum(max(0, diff[i] - diff[i-1])) for i from 1 to n-1.
        # Explanation:
        # - abs(diff[0]) operations are needed to "start" from zero difference.
        # - For each i from 1 to n-1, if diff[i] > diff[i-1],
        #   the extra (diff[i]-diff[i-1]) must be added by extra operations.
        
        operations = abs(diff[0])
        for i in range(1, n):
            if diff[i] - diff[i-1] > 0:
                operations += diff[i] - diff[i-1]
        
        return operations