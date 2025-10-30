from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum operations needed to make the median of nums equal to k.

        An operation consists of increasing or decreasing any element by 1.
        The median of an array is defined as the middle element when sorted.
        If there are two choices for a median (even length), the larger is taken.

        Args:
            nums: A list of integers.
            k: A non-negative integer target median value.

        Returns:
            The minimum number of operations required.
        """
        # Sort the array to easily identify the median element and elements relative to it.
        nums.sort()
        
        n = len(nums)
        # Determine the median index based on the problem's definition.
        # For 0-indexed array of length n, the middle elements are at indices
        # (n-1)//2 and n//2. The problem states the larger of the two is taken for even length,
        # and the single middle element for odd length.
        # This maps to index n // 2 for both odd and even n in a 0-indexed array.
        # Example: n=5 -> median index 5//2 = 2. elements [0, 1, 2, 3, 4], median at 2.
        # Example: n=6 -> median index 6//2 = 3. elements [0, 1, 2, 3, 4, 5], middle are 2 and 3, larger is at 3.
        median_index = n // 2
        
        operations = 0
        
        current_median_value = nums[median_index]

        # The element at median_index in the sorted array must become k.
        # Operations are needed for nums[median_index] and potentially other elements
        # to ensure k is the median in the sorted modified array.

        if current_median_value < k:
            # The current median is less than k.
            # We need to increase nums[median_index] to k, costing k - nums[median_index].
            # To ensure k is the median, all elements from median_index onwards
            # in the sorted modified array must be >= k.
            # We iterate from the median element to the right (inclusive).
            # For each element nums[i] in this range (i >= median_index) that is currently < k,
            # we must increase it to at least k. The minimum cost is k - nums[i].
            # Elements nums[i] for i < median_index are <= current_median_value < k,
            # so they are already <= k, satisfying the potential requirement for the left side.
            for i in range(median_index, n):
                if nums[i] < k:
                    operations += k - nums[i]
                else:
                    # Optimization: Since the array is sorted, if nums[i] >= k, all subsequent
                    # elements nums[j] (j > i) will also be >= k.
                    # These elements are on the right side (>= median_index) and already satisfy
                    # the >= k condition. We don't need to check or modify them.
                    break 
                    
        elif current_median_value > k:
            # The current median is greater than k.
            # We need to decrease nums[median_index] to k, costing nums[median_index] - k.
            # To ensure k is the median, all elements up to median_index
            # in the sorted modified array must be <= k.
            # We iterate from the median element to the left (inclusive).
            # For each element nums[i] in this range (i <= median_index) that is currently > k,
            # we must decrease it to at most k. The minimum cost is nums[i] - k.
            # Elements nums[i] for i > median_index are >= current_median_value > k,
            # so they are already > k, satisfying the potential requirement for the right side.
            for i in range(median_index, -1, -1): # Iterate from median_index down to 0
                if nums[i] > k:
                    operations += nums[i] - k
                else:
                    # Optimization: Since the array is sorted, if nums[i] <= k, all previous
                    # elements nums[j] (j < i) will also be <= k.
                    # These elements are on the left side (<= median_index) and already satisfy
                    # the <= k condition. We don't need to check or modify them.
                    break
        
        # If current_median_value == k, operations is initialized to 0 and remains 0, which is correct.
        
        return operations