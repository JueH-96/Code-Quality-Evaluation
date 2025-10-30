class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Initialize the array with the first element as x
        nums = [x]
        
        # Generate the rest of the elements
        for i in range(1, n):
            # The next element should be greater than the previous one
            # and should maintain the bitwise AND condition
            nums.append(nums[-1] | (1 << i))
        
        # Return the last element of the array
        return nums[-1]