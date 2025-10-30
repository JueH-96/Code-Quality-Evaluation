from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Step 1: Calculate the sum of the longest sequential prefix.
        # The first element nums[0] always forms a sequential prefix.
        sum_of_longest_sequential_prefix = nums[0]
        
        # Iterate from the second element to check if the sequence continues.
        # The loop stops either at the end of the array or when the sequential
        # property (nums[i] == nums[i-1] + 1) is broken.
        for i in range(1, len(nums)):
            # If the current number continues the sequence, add it to the sum.
            if nums[i] == nums[i-1] + 1:
                sum_of_longest_sequential_prefix += nums[i]
            else:
                # If the sequence is broken, the longest sequential prefix
                # starting from nums[0] has been found. Exit the loop.
                break
        
        # Step 2: Find the smallest integer x that is >= the calculated sum
        #         and is not present in the original nums array.
        
        # Convert the nums array to a set for efficient O(1) average-time lookups.
        nums_set = set(nums)
        
        # Start checking from the sum of the longest sequential prefix.
        x = sum_of_longest_sequential_prefix
        
        # Increment x iteratively until a value not found in nums_set is encountered.
        while True:
            if x not in nums_set:
                # This 'x' is the smallest integer that meets both conditions.
                return x
            else:
                # If 'x' is in the set, try the next integer.
                x += 1