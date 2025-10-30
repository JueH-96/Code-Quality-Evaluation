import math # This line is not needed and can be removed.
from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of right shifts required to sort the array nums.

        A right shift moves the element at index i to index (i + 1) % n.
        An array can be sorted by right shifts if and only if it is a rotation
        of its sorted version. This property implies that when traversing the
        array circularly, there should be at most one point where the order
        decreases (i.e., nums[i] > nums[(i + 1) % n]).

        Args:
            nums: A list of distinct positive integers.

        Returns:
            The minimum number of right shifts required to sort the array,
            or -1 if it's impossible to sort using right shifts.
        """
        n = len(nums)
        
        # Base case: Arrays of size 0 or 1 are considered already sorted.
        # 0 shifts are needed.
        if n <= 1:
            return 0

        # Count the number of "break points" where the ascending order is violated.
        # A break point occurs at index `i` if nums[i] > nums[(i + 1) % n].
        # We also need to store the index of the break point if only one exists.
        breaks = 0
        break_index = -1 

        # Iterate through the array, including the wrap-around comparison
        # from the last element (index n-1) to the first element (index 0).
        for i in range(n):
            current_element = nums[i]
            # Calculate the index of the next element in a circular manner.
            next_index = (i + 1) % n 
            next_element = nums[next_index]

            # Check if the current element is greater than the next element.
            if current_element > next_element:
                breaks += 1
                # Record the index `i` where the break occurs.
                break_index = i 
                
                # Optimization: If we detect more than one break point early,
                # we know it's impossible to sort by right shifts. We could
                # return -1 immediately. However, completing the loop is also fine
                # given the small constraint n <= 100.
                # if breaks > 1:
                #     return -1

        # After checking all elements, analyze the number of breaks found.
        if breaks == 0:
            # If there are no breaks, it means nums[0] <= nums[1] <= ... <= nums[n-1] <= nums[0].
            # Since the problem states elements are distinct, this implies strict inequalities:
            # nums[0] < nums[1] < ... < nums[n-1] < nums[0].
            # This forms a contradiction for n > 1.
            # The only way breaks == 0 is if the loop condition was never met.
            # This occurs for n=1 (handled by the base case) or if the array is already sorted
            # and nums[n-1] <= nums[0] holds. With distinct elements, this means the array
            # is sorted: nums[0] < nums[1] < ... < nums[n-1].
            # In this case, 0 shifts are required.
            return 0
            
        elif breaks == 1:
            # If there is exactly one break point, the array is a single rotation
            # of a sorted array. The minimum element of the sorted version must be
            # located right after the break point in the current `nums` array.
            # The index of this minimum element is (break_index + 1) % n.
            min_element_index = (break_index + 1) % n
            
            # The number of right shifts required is the number of steps needed
            # to move the element currently at `min_element_index` to index 0.
            # If `min_element_index` is 0, the element is already at the start,
            # meaning the array was already sorted, and 0 shifts are needed.
            # If `min_element_index` is `p > 0`, we need `n - p` shifts.
            # The formula `(n - min_element_index) % n` correctly calculates the
            # number of shifts for both cases (p=0 and p>0).
            shifts_required = (n - min_element_index) % n
            return shifts_required
            
        else: # breaks > 1
            # If there are more than one break points, the array's structure
            # is not a simple rotation of a sorted sequence. It cannot be
            # sorted using only right shifts.
            return -1