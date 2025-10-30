from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute_gcd(arr):
            if not arr:
                return 0
            current_gcd = arr[0]
            for num in arr[1:]:
                current_gcd = math.gcd(current_gcd, num)
                if current_gcd == 1:
                    break  # GCD cannot be smaller than 1
            return current_gcd
        
        def compute_lcm(arr):
            if not arr:
                return 0
            current_lcm = arr[0]
            for num in arr[1:]:
                current_lcm = (current_lcm * num) // math.gcd(current_lcm, num)
            return current_lcm
        
        max_score = 0
        # Calculate the original array's score
        original_gcd = compute_gcd(nums)
        original_lcm = compute_lcm(nums)
        max_score = original_gcd * original_lcm
        
        # Check each possible subset with one element removed
        for i in range(len(nums)):
            subset = nums[:i] + nums[i+1:]
            subset_gcd = compute_gcd(subset)
            subset_lcm = compute_lcm(subset)
            current_score = subset_gcd * subset_lcm
            if current_score > max_score:
                max_score = current_score
        
        return max_score