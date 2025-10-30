from math import gcd
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # Helper to compute lcm of two numbers
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)
        
        # Helper to compute LCM and GCD for a list of numbers
        def compute_lcm_and_gcd(arr: List[int]) -> (int, int):
            if not arr:
                return 0, 0
            curr_gcd = arr[0]
            curr_lcm = arr[0]
            for num in arr[1:]:
                curr_gcd = gcd(curr_gcd, num)
                curr_lcm = lcm(curr_lcm, num)
            return curr_lcm, curr_gcd
        
        n = len(nums)
        max_score = 0
        
        # Consider no removal (use full array)
        full_lcm, full_gcd = compute_lcm_and_gcd(nums)
        max_score = max(max_score, full_lcm * full_gcd)
        
        # Consider removal of one element
        for i in range(n):
            # Form a new array by removing the element at index i
            new_nums = nums[:i] + nums[i+1:]
            if new_nums:  # Only compute if non-empty; factor score of empty is defined 0.
                curr_lcm, curr_gcd = compute_lcm_and_gcd(new_nums)
                score = curr_lcm * curr_gcd
                max_score = max(max_score, score)
        
        return max_score