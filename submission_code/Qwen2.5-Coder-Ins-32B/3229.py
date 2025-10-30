from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            s = str(x)
            return s == s[::-1]
        
        def find_closest_palindrome(x):
            if is_palindrome(x):
                return x
            L = x - 1
            R = x + 1
            while True:
                if is_palindrome(L):
                    return L
                if is_palindrome(R):
                    return R
                L -= 1
                R += 1
        
        median = sorted(nums)[len(nums) // 2]
        p1 = find_closest_palindrome(median)
        p2 = find_closest_palindrome(median - 1)
        p3 = find_closest_palindrome(median + 1)
        
        def calculate_cost(target):
            return sum(abs(num - target) for num in nums)
        
        return min(calculate_cost(p1), calculate_cost(p2), calculate_cost(p3))