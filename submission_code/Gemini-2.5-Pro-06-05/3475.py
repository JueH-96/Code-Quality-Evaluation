from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum operations to make all elements in a binary array 1.
        An operation consists of flipping 3 consecutive elements.
        """
        n = len(nums)
        operations = 0
        
        # We iterate from left to right.
        # For each element nums[i], if its current value is 0, we must perform an
        # operation starting at index i. This is the only operation that can make
        # nums[i] equal to 1 without changing any of the previous elements
        # nums[0...i-1], which we have already processed.
        
        # The last possible starting index for a 3-element flip is n-3.
        # The loop range goes from i=0 to n-3.
        for i in range(n - 2):
            if nums[i] == 0:
                # We must flip the triplet starting at i to make nums[i] become 1.
                operations += 1
                
                # Apply the flip to the three consecutive elements.
                # It's crucial to update all three to propagate the state correctly for
                # subsequent iterations.
                nums[i] = 1
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
        
        # After the loop, by design, elements nums[0] through nums[n-3] are 1.
        # We now check the last two elements. If either is 0, it's impossible to
        # make the whole array 1s, as no more operations can affect them without
        # going out of bounds.
        if nums[n-2] == 0 or nums[n-1] == 0:
            return -1
        
        return operations