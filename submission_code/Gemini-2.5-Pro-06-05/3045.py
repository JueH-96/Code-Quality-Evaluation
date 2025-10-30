from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of right shifts to sort an array.

        The solution is based on the property that an array that can be sorted 
        by cyclic shifts must be a rotated version of a sorted array. Such an array
        (with distinct elements) has exactly one "drop" point where nums[i] > nums[i+1]
        when considered cyclically.

        The algorithm finds this single drop point. The element immediately after the 
        drop is the minimum element of the array. The number of right shifts required 
        is the number needed to move this minimum element to the first position (index 0).
        """
        n = len(nums)
        if n <= 1:
            return 0

        # We are looking for a single "drop" in the cyclic sequence.
        # A drop is an index `i` where nums[i] > nums[(i + 1) % n].
        drop_index = -1
        
        for i in range(n):
            # Compare current element with the next, wrapping around for the last element.
            if nums[i] > nums[(i + 1) % n]:
                # If we've already found a drop, this means there are at least two.
                # Such an array cannot be sorted by rotation.
                if drop_index != -1:
                    return -1
                drop_index = i
        
        # If no drop was found after a full cyclic pass, it implies nums[0]<=nums[1]<=...<=nums[n-1]<=nums[0].
        # With distinct integers, this is impossible for n > 1. 
        # For a valid cyclically sorted array (n > 1), drop_index will always be set.
        
        # The drop is at `drop_index`. The smallest element is at the next position.
        min_element_index = (drop_index + 1) % n
        
        # If the minimum element is already at index 0, the array is sorted.
        # This happens when the drop is at the last element (i.e., drop_index = n - 1).
        if min_element_index == 0:
            return 0
        
        # Otherwise, we need to shift the array right until the minimum element at `min_element_index`
        # reaches index 0. The number of right shifts required to move an element from 
        # index `k` to index `0` is `n - k`.
        return n - min_element_index