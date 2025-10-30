from math import gcd

class Solution:
    def maxScore(self, nums: list[int]) -> int:
        n = len(nums)
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def compute_score(arr):
            if not arr:
                return 0
            current_gcd = arr[0]
            current_lcm = arr[0]
            for i in range(1, len(arr)):
                current_gcd = gcd(current_gcd, arr[i])
                current_lcm = lcm(current_lcm, arr[i])
            return current_gcd * current_lcm
        
        best_score = compute_score(nums)
        
        for i in range(n):
            sub_arr = nums[:i] + nums[i+1:]
            score = compute_score(sub_arr)
            if score > best_score:
                best_score = score
        
        return best_score