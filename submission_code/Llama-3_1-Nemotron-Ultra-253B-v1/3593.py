from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute_gcd(arr):
            if not arr:
                return 0
            current = arr[0]
            for num in arr[1:]:
                current = math.gcd(current, num)
                if current == 1:
                    break
            return current
        
        def compute_lcm(arr):
            if not arr:
                return 0
            current = arr[0]
            for num in arr[1:]:
                current = (current * num) // math.gcd(current, num)
            return current
        
        original_gcd = compute_gcd(nums)
        original_lcm = compute_lcm(nums)
        max_score = original_gcd * original_lcm
        
        for i in range(len(nums)):
            new_arr = nums[:i] + nums[i+1:]
            if not new_arr:
                current_score = 0
            else:
                gcd_val = compute_gcd(new_arr)
                lcm_val = compute_lcm(new_arr)
                current_score = gcd_val * lcm_val
            if current_score > max_score:
                max_score = current_score
        
        return max_score