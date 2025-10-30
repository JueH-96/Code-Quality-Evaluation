class Solution:
    def minOrAfterOperations(self, nums: list[int], k: int) -> int:
        # If k is 0, we cannot perform any operations, so we return the bitwise OR of all elements.
        if k == 0:
            return reduce(lambda x, y: x | y, nums)
        
        # If k is greater than or equal to the length of nums minus 1, we can reduce the array to a single element.
        if k >= len(nums) - 1:
            return reduce(lambda x, y: x & y, nums)
        
        # Otherwise, we perform the operations to minimize the OR value.
        # We start by calculating the OR of all elements.
        total_or = reduce(lambda x, y: x | y, nums)
        
        # We iterate over the array and calculate the effect of each AND operation on the total OR.
        for i in range(len(nums) - 1):
            # Calculate the OR value if we perform the AND operation at index i.
            new_or = total_or & ~(nums[i] ^ nums[i + 1])
            # If the new OR value is less than the current total OR, we update the total OR.
            if new_or < total_or:
                total_or = new_or
                # We decrement k because we have used one operation.
                k -= 1
                # If k reaches 0, we have used all operations and return the current total OR.
                if k == 0:
                    return total_or
        
        # If we have operations left, we return the current total OR.
        return total_or

# Helper function to use reduce from functools
from functools import reduce