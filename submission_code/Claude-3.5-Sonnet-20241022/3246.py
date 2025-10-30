class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Count numbers that are even (have trailing zeros)
        even_count = sum(1 for num in nums if num % 2 == 0)
        
        # If we have 2 or more even numbers, their OR will have trailing zeros
        return even_count >= 2