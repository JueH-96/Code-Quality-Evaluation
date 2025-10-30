from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)

        # An array with 1 element is considered sorted and requires 0 shifts.
        if n == 1:
            return 0
        
        # Step 1: Find the minimum element and its index in the original array.
        # This minimum element, when brought to the first position, should ideally
        # make the array sorted.
        min_val = nums[0]
        min_idx = 0
        for i in range(1, n):
            if nums[i] < min_val:
                min_val = nums[i]
                min_idx = i
        
        # Step 2: Construct a new array by rotating `nums` such that `min_val`
        # is at the beginning. This represents the state of the array after
        # the required number of right shifts.
        # If min_val is at `min_idx`, then `n - min_idx` right shifts are
        # needed to bring it to index 0.
        # For example, if nums = [3,4,5,1,2] and min_idx = 3 (value 1),
        # temp_nums will be [1,2,3,4,5].
        temp_nums = nums[min_idx:] + nums[:min_idx]

        # Step 3: Check if this `temp_nums` array is sorted in ascending order.
        # An array is sorted if each element is less than its successor.
        is_sorted = True
        for i in range(n - 1):
            if temp_nums[i] > temp_nums[i+1]:
                is_sorted = False
                break
        
        # Step 4: Determine the result based on the sortedness check.
        if is_sorted:
            # If the constructed array is sorted, return the number of shifts.
            # The number of right shifts required is `(n - min_idx)`.
            # We use `% n` to correctly handle the case where `min_idx` is 0
            # (array already sorted), which results in 0 shifts.
            return (n - min_idx) % n
        else:
            # If the constructed array is not sorted, it means the original array
            # cannot be sorted by any number of right shifts.
            return -1