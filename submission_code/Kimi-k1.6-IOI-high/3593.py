from typing import List
from math import gcd

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute_score(arr):
            if not arr:
                return 0
            current_gcd = arr[0]
            current_lcm = arr[0]
            for num in arr[1:]:
                current_gcd = gcd(current_gcd, num)
                current_lcm = current_lcm * num // gcd(current_lcm, num)
            return current_gcd * current_lcm
        
        max_score = compute_score(nums)
        n = len(nums)
        for i in range(n):
            new_arr = nums[:i] + nums[i+1:]
            current_score = compute_score(new_arr)
            if current_score > max_score:
                max_score = current_score
        return max_score