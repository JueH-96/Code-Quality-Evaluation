class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Initialize a variable to keep track of the bitwise OR of all elements
        all_or = 0
        
        # Calculate the bitwise OR of all elements in the array
        for num in nums:
            all_or |= num
        
        # Check if the bitwise OR of all elements has at least one trailing zero
        # If it does, return True, otherwise return False
        return all_or & (all_or + 1) == 0