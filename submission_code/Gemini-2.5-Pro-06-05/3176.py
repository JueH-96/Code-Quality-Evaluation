from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        """
        Finds the minimum sum of a mountain triplet in an array.

        A mountain triplet (i, j, k) satisfies i < j < k, nums[i] < nums[j], and nums[k] < nums[j].
        """
        n = len(nums)
        min_sum = float('inf')
        
        # Iterate through each potential peak `nums[j]`.
        # A peak must have elements to its left and right, so j's range is [1, n-2].
        for j in range(1, n - 1):
            
            # For a given peak `nums[j]`, we need to find the smallest element
            # to its left (`nums[i]`) and the smallest to its right (`nums[k]`)
            # to form a potential mountain triplet (i, j, k).
            
            min_left = float('inf')
            # Find the minimum value in the subarray to the left of j.
            for i in range(j):
                min_left = min(min_left, nums[i])
            
            min_right = float('inf')
            # Find the minimum value in the subarray to the right of j.
            for k in range(j + 1, n):
                min_right = min(min_right, nums[k])
                
            # A mountain triplet requires nums[i] < nums[j] and nums[k] < nums[j].
            # If the minimums on the left and right satisfy this, a valid triplet exists.
            if nums[j] > min_left and nums[j] > min_right:
                # The sum of this triplet is min_left + nums[j] + min_right.
                # We update our overall minimum sum.
                current_sum = min_left + nums[j] + min_right
                min_sum = min(min_sum, current_sum)
                
        # If min_sum remains at its initial large value, no mountain triplet was found.
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum