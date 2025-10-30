from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Helper to compute the maximum digit of a number
        def max_digit(num: int) -> int:
            m = 0
            while num:
                m = max(m, num % 10)
                num //= 10
            return m
        
        # Group numbers by their maximum digit
        groups = {}
        for x in nums:
            d = max_digit(x)
            if d not in groups:
                groups[d] = []
            groups[d].append(x)
        
        # For each group with at least two members, compute the best pair sum
        best = -1
        for lst in groups.values():
            if len(lst) >= 2:
                lst.sort(reverse=True)
                s = lst[0] + lst[1]
                if s > best:
                    best = s
        
        return best