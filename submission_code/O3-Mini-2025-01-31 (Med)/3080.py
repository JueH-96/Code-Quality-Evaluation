from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # First, compute the global bitwise AND of the entire array.
        global_and = nums[0]
        for x in nums[1:]:
            global_and &= x
        
        # If the global AND is not 0, then any partition will have a sum of scores 
        # at least equal to global_and, and splitting further won't reduce the sum.
        # So the only optimal split is to keep the array whole.
        if global_and != 0:
            return 1
        
        # Otherwise, global_and equals 0.
        # We want to split the array into as many subarrays as possible all yielding score 0.
        # We can greedily accumulate the bitwise AND from the left and whenever it becomes 0,
        # that subarray's score is 0 and we can make a split here.
        count = 0
        # Since nums[i] are non-negative and can go up to 10^6 (around 20 bits),
        # we initialize current to a number with all bits set in a sufficiently large range.
        # Using (1 << 31) - 1 is safe.
        current = (1 << 31) - 1  
        for x in nums:
            current &= x
            if current == 0:
                count += 1
                # Reset current value for the next subarray accumulation.
                current = (1 << 31) - 1
        
        return count