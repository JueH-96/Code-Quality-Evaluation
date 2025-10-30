from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # last_group_sum stores the sum of the most recently finalized group.
        # This is the value that the next group's sum must be greater than or equal to.
        # Initialize to 0. Since all nums[i] >= 1, the first group's sum will always
        # be >= 0, allowing the first group to be finalized.
        last_group_sum = 0
        
        # length stores the number of groups successfully formed so far.
        # This is the length of the non-decreasing array being built.
        length = 0
        
        # current_group_sum accumulates the sum of elements for the group currently being formed.
        # When this sum is finalized as a group, it is reset to 0 for the next group.
        current_group_sum = 0

        # Iterate through each number in the input array from left to right.
        for x in nums:
            # Add the current number to the sum of the potential next group.
            current_group_sum += x
            
            # Check if the current accumulated sum can form the next non-decreasing group.
            # A new group can be formed if its sum is greater than or equal to the sum
            # of the previous group (last_group_sum).
            # According to the greedy strategy, we finalize the current group as soon as
            # this condition is met. This is because finalizing now uses the minimum
            # possible sum starting from the current position that satisfies the
            # non-decreasing requirement (since current_group_sum is the smallest prefix
            # sum >= last_group_sum starting from the end of the previous group).
            # Minimizing the current group's sum leaves the maximum possible remaining
            # elements for subsequent groups, increasing the chances of forming more groups
            # and thus maximizing the total length.
            if current_group_sum >= last_group_sum:
                # Finalize the current group:
                # 1. The sum of the currently accumulated group becomes the reference sum
                #    for the next group.
                last_group_sum = current_group_sum
                
                # 2. We have successfully formed one more non-decreasing group.
                length += 1
                
                # 3. Reset the accumulator for the next group. We start forming the
                #    next group using the elements immediately following the current one.
                current_group_sum = 0
            # Else (current_group_sum < last_group_sum), we cannot finalize the current group yet
            # as doing so would violate the non-decreasing property (the current group's sum
            # would be less than the previous group's sum). We continue accumulating
            # more elements into current_group_sum in the next iteration, hoping that
            # adding subsequent elements will eventually make current_group_sum
            # large enough to satisfy the condition.

        # After processing all elements, the total number of groups finalized
        # represents the maximum length of the non-decreasing array that can be formed
        # by partitioning the original array according to the greedy strategy.
        return length