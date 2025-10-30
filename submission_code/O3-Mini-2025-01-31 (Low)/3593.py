from math import gcd
from functools import reduce
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # Helper to compute gcd for a list of numbers
        def compute_gcd(arr: List[int]) -> int:
            return reduce(gcd, arr) if arr else 0
        
        # Helper to compute lcm of two numbers
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)
        
        # Helper to compute lcm for a list of numbers
        def compute_lcm(arr: List[int]) -> int:
            return reduce(lcm, arr) if arr else 0
        
        n = len(nums)
        # Option 1: no element is removed
        max_score = 0
        # If the entire array is non-empty, compute its factor score.
        if n > 0:
            full_gcd = compute_gcd(nums)
            full_lcm = compute_lcm(nums)
            max_score = full_gcd * full_lcm
        
        # Option 2: remove one element (at most one)
        for i in range(n):
            # Create a new list with the ith element removed
            new_nums = nums[:i] + nums[i+1:]
            if not new_nums:
                # Factor score for empty array is 0.
                continue
            current_gcd = compute_gcd(new_nums)
            current_lcm = compute_lcm(new_nums)
            score = current_gcd * current_lcm
            if score > max_score:
                max_score = score
        
        return max_score