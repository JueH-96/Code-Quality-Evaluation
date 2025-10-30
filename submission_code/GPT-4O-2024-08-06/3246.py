class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Check if there are at least two even numbers
        even_count = sum(1 for num in nums if num % 2 == 0)
        
        # If there are at least two even numbers, we can select them to have trailing zeros
        if even_count >= 2:
            return True
        
        # Check if there is at least one even number and one odd number
        if even_count >= 1 and even_count < len(nums):
            return True
        
        # If all numbers are odd, it's not possible to have trailing zeros
        return False