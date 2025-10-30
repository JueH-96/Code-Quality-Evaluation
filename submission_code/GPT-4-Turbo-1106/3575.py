class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Initialize the OR values for the first and second halves
        first_half_or = 0
        second_half_or = 0
        
        # Calculate the OR value for the first half
        for i in range(k):
            first_half_or |= nums[i]
        
        # Calculate the OR value for the second half
        for i in range(k, 2 * k):
            second_half_or |= nums[i]
        
        # Return the XOR of the two halves
        return first_half_or ^ second_half_or