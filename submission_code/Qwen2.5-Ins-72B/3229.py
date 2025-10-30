from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        def find_closest_palindromes(x):
            lower = upper = x
            while not is_palindrome(lower):
                lower -= 1
            while not is_palindrome(upper):
                upper += 1
            return lower, upper
        
        def calculate_cost(target):
            return sum(abs(num - target) for num in nums)
        
        nums.sort()
        n = len(nums)
        median = nums[n // 2]
        
        lower_palindrome, upper_palindrome = find_closest_palindromes(median)
        
        return min(calculate_cost(lower_palindrome), calculate_cost(upper_palindrome))