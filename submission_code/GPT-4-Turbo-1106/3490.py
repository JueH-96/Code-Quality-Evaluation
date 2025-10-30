class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the number of even and odd numbers
        even_count = sum(num % 2 == 0 for num in nums)
        odd_count = len(nums) - even_count
        
        # The longest valid subsequence will be the one with all even or all odd numbers
        # Since the sum of two even or two odd numbers is always even
        return max(even_count, odd_count)