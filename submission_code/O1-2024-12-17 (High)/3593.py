from math import gcd
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # Function to compute GCD of a list of numbers
        def compute_gcd(arr):
            if not arr:
                return 0  # gcd of empty array is not well-defined, but we'll use 0
            current_gcd = arr[0]
            for num in arr[1:]:
                current_gcd = gcd(current_gcd, num)
            return current_gcd
        
        # Function to compute LCM of two numbers
        def lcm(a, b):
            return a * b // gcd(a, b) if a and b else 0
        
        # Function to compute LCM of a list of numbers
        def compute_lcm(arr):
            if not arr:
                return 0
            current_lcm = 1
            for num in arr:
                current_lcm = lcm(current_lcm, num)
            return current_lcm
        
        # Edge case: if there's only one element, answer is that number squared or 0 after removal
        if len(nums) == 1:
            return nums[0] * nums[0]  # or 0, whichever is maximum (the single-element product wins)
        
        # Compute gcd and lcm of the entire array
        gcd_all = compute_gcd(nums)
        lcm_all = compute_lcm(nums)
        best_score = gcd_all * lcm_all
        
        # Try removing each element
        for i in range(len(nums)):
            # Build a list without the i-th element
            reduced = nums[:i] + nums[i+1:]
            # Compute gcd and lcm of the reduced list
            gcd_reduced = compute_gcd(reduced)
            lcm_reduced = compute_lcm(reduced)
            best_score = max(best_score, gcd_reduced * lcm_reduced)
        
        return best_score