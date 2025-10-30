from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0

        # Iterate through the array up to n-3.
        # The loop range `n - 2` means `i` will take values from `0` to `n-3` inclusive.
        # For the last possible operation starting at `i = n-3`, it affects `nums[n-3]`, `nums[n-2]`, `nums[n-1]`.
        for i in range(n - 2):
            # If the current element nums[i] is 0, we must flip it to 1.
            # The greedy choice is to perform an operation starting at index i.
            # This is because any operation starting before i would affect elements to the left of i,
            # which we have already processed and ensured are 1s.
            if nums[i] == 0:
                operations += 1
                # Flip the three consecutive elements starting from i.
                # A flip means changing 0 to 1, and 1 to 0. This can be done by 1 - value.
                nums[i] = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
        
        # After the loop, all elements from nums[0] to nums[n-3] are guaranteed to be 1s.
        # This is a direct consequence of the greedy strategy.

        # Now, we must check the last two elements, nums[n-2] and nums[n-1].
        # If either of these is 0, it means it's impossible to make all elements 1.
        # This is because there are no further operations that can be performed to affect these
        # elements without going out of bounds or affecting elements to their left (which are already 1s).
        if nums[n-2] == 0 or nums[n-1] == 0:
            return -1
        
        # If we reach here, it means all elements in the array are now 1s.
        # Return the total number of operations performed.
        return operations