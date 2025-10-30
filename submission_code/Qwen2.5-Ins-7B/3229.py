from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]
        
        def closest_palindrome(num):
            if is_palindrome(num):
                return num
            lower, upper = num, num + 1
            while not is_palindrome(lower):
                lower -= 1
            while not is_palindrome(upper):
                upper += 1
            return lower if num - lower <= upper - num else upper
        
        nums.sort()
        median = nums[len(nums) // 2]
        closest_med = closest_palindrome(median)
        closest_med_next = closest_palindrome(median + 1)
        
        cost_med = sum(abs(num - closest_med) for num in nums)
        cost_med_next = sum(abs(num - closest_med_next) for num in nums)
        
        return min(cost_med, cost_med_next)