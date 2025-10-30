from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # 1. Sort the array in non-decreasing order.
        # This step is crucial because the median is defined on a sorted array.
        nums.sort()
        
        # 2. Determine the median index based on the problem's definition.
        # For an array of length n (0-indexed), the median is at index n // 2.
        # This handles both odd and even length arrays according to the "larger of two" rule.
        # Examples:
        #   n=5: indices [0, 1, 2, 3, 4]. 5 // 2 = 2. Median is nums[2].
        #   n=6: indices [0, 1, 2, 3, 4, 5]. 6 // 2 = 3. Median is nums[3] (the larger of middle two).
        median_idx = n // 2
        
        operations = 0
        
        # 3. Calculate operations needed for the median element itself.
        # The element at nums[median_idx] MUST be changed to k.
        operations += abs(nums[median_idx] - k)
        
        # Now, conceptually, nums[median_idx] is k. To ensure k remains the median
        # at this position in the sorted array, we must satisfy two conditions:
        # - All elements to its left (indices < median_idx) must be less than or equal to k.
        # - All elements to its right (indices > median_idx) must be greater than or equal to k.
        
        # 4. Calculate operations for elements to the left of the median_idx.
        # Iterate from median_idx - 1 down to 0.
        for i in range(median_idx - 1, -1, -1):
            if nums[i] > k:
                # If an element to the left is greater than k, it violates the condition (nums[i] <= k).
                # To satisfy this with minimum operations, we must decrease nums[i] to k.
                operations += (nums[i] - k)
            else:
                # If nums[i] is already less than or equal to k, it satisfies the condition.
                # Since the array is sorted, all elements at indices smaller than i (further left)
                # will also be less than or equal to nums[i], and thus less than or equal to k.
                # Therefore, no further operations are needed for elements to the left.
                break 
        
        # 5. Calculate operations for elements to the right of the median_idx.
        # Iterate from median_idx + 1 up to n - 1.
        for i in range(median_idx + 1, n):
            if nums[i] < k:
                # If an element to the right is less than k, it violates the condition (nums[i] >= k).
                # To satisfy this with minimum operations, we must increase nums[i] to k.
                operations += (k - nums[i])
            else:
                # If nums[i] is already greater than or equal to k, it satisfies the condition.
                # Since the array is sorted, all elements at indices larger than i (further right)
                # will also be greater than or equal to nums[i], and thus greater than or equal to k.
                # Therefore, no further operations are needed for elements to the right.
                break 
                
        return operations