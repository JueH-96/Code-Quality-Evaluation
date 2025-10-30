from math import gcd
from typing import List

class Solution:
    # Helper function to compute least common multiple of two numbers.
    def lcm(self, a: int, b: int) -> int:
        return a * b // gcd(a, b)
    
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        # Edge case: if the list is empty (shouldn't happen per constraints)
        if n == 0:
            return 0
        
        # If there's only one element, best is to use it (without removal), because removing it gives 0.
        if n == 1:
            return nums[0] * nums[0]
        
        # Precompute prefix gcd and prefix lcm arrays
        prefix_gcd = [0] * n
        prefix_lcm = [0] * n
        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = gcd(prefix_gcd[i - 1], nums[i])
            prefix_lcm[i] = self.lcm(prefix_lcm[i - 1], nums[i])
        
        # Precompute suffix gcd and suffix lcm arrays
        suffix_gcd = [0] * n
        suffix_lcm = [0] * n
        suffix_gcd[-1] = nums[-1]
        suffix_lcm[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            suffix_gcd[i] = gcd(suffix_gcd[i + 1], nums[i])
            suffix_lcm[i] = self.lcm(suffix_lcm[i + 1], nums[i])
        
        # Candidate when using no removal
        full_gcd = prefix_gcd[-1]
        full_lcm = prefix_lcm[-1]
        max_score = full_gcd * full_lcm
        
        # Try removing each element once
        for i in range(n):
            # Compute gcd for the array without the element at index i
            if i == 0:
                curr_gcd = suffix_gcd[1]
                curr_lcm = suffix_lcm[1]
            elif i == n - 1:
                curr_gcd = prefix_gcd[n - 2]
                curr_lcm = prefix_lcm[n - 2]
            else:
                curr_gcd = gcd(prefix_gcd[i - 1], suffix_gcd[i + 1])
                # To compute lcm from two numbers: if one of them is 0, the lcm is the other added value.
                # But here neither prefix_lcm[i-1] nor suffix_lcm[i+1] will be 0.
                curr_lcm = self.lcm(prefix_lcm[i - 1], suffix_lcm[i + 1])
            
            max_score = max(max_score, curr_gcd * curr_lcm)
        
        return max_score