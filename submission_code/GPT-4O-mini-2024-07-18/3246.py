class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Count how many numbers are even
        even_count = sum(1 for num in nums if num % 2 == 0)
        
        # If there are at least two even numbers, we can form a pair with trailing zeros
        return even_count >= 2