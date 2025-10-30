class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = sum(1 for num in nums if num % 2 == 0)
        odd_count = len(nums) - even_count
        
        # The longest valid subsequence can be formed by taking all even or all odd numbers
        return len(nums) if even_count == 0 or odd_count == 0 else len(nums) - abs(even_count - odd_count)