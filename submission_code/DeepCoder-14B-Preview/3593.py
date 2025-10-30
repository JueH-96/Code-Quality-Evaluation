import math
from functools import reduce

class Solution:
    def maxScore(self, nums: list[int]) -> int:
        if not nums:
            return 0
        
        def compute_gcd(arr):
            return reduce(math.gcd, arr)
        
        def compute_lcm(arr):
            if not arr:
                return 0
            current_lcm = arr[0]
            for num in arr[1:]:
                current_gcd = math.gcd(current_lcm, num)
                current_lcm = (current_lcm * num) // current_gcd
                if current_lcm == 0:
                    return 0
            return current_lcm
        
        original_gcd = compute_gcd(nums)
        original_lcm = compute_lcm(nums)
        max_score = original_gcd * original_lcm
        
        for i in range(len(nums)):
            new_nums = nums[:i] + nums[i+1:]
            if not new_nums:
                current_score = 0
            else:
                current_gcd = compute_gcd(new_nums)
                current_lcm = compute_lcm(new_nums)
                current_score = current_gcd * current_lcm
            if current_score > max_score:
                max_score = current_score
        
        return max_score