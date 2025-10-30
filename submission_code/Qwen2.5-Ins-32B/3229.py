from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        def closest_palindrome(x):
            lower, upper = x, x
            while not is_palindrome(lower):
                lower -= 1
            while not is_palindrome(upper):
                upper += 1
            return lower, upper
        
        def cost_to_palindrome(target):
            return sum(abs(num - target) for num in nums)
        
        nums.sort()
        median = nums[len(nums) // 2]
        lower_palindrome, upper_palindrome = closest_palindrome(median)
        
        cost1 = cost_to_palindrome(lower_palindrome)
        cost2 = cost_to_palindrome(upper_palindrome)
        
        return min(cost1, cost2)