class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Initialize the first element of the array to be the smallest possible value
        nums = [x]
        
        # Construct the rest of the array
        for i in range(1, n):
            # Find the smallest possible value for the next element
            # that is greater than the previous element and maintains the bitwise AND
            next_val = nums[-1] + 1
            while (next_val & x) != x:
                next_val += 1
            nums.append(next_val)
        
        # Return the last element of the array
        return nums[-1]