from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum operations to make the median of an array equal to k.
        """
        nums.sort()
        
        n = len(nums)
        mid_index = n // 2
        
        operations = 0
        
        # According to the problem, the median is the element at index n // 2
        # of the sorted array. This holds for both odd and even length arrays
        # based on the "larger of the two" rule for even lengths.
        
        # Case 1: The current median is less than k.
        # To make the median k, all elements from the median position onwards
        # must be at least k.
        if nums[mid_index] < k:
            # We sum the operations needed to raise all elements in the
            # upper half (from median onwards) that are less than k, up to k.
            for i in range(mid_index, n):
                if nums[i] < k:
                    operations += k - nums[i]
                else:
                    # Since the array is sorted, if nums[i] is already >= k,
                    # all subsequent elements will be too. We can stop early.
                    break
        
        # Case 2: The current median is greater than k.
        # To make the median k, all elements up to the median position
        # must be at most k.
        elif nums[mid_index] > k:
            # We sum the operations needed to lower all elements in the
            # lower half (up to median) that are greater than k, down to k.
            for i in range(mid_index, -1, -1):
                if nums[i] > k:
                    operations += nums[i] - k
                else:
                    # Since the array is sorted, if nums[i] is already <= k,
                    # all preceding elements will be too. We can stop early.
                    break
            
        return operations