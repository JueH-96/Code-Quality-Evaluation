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
                current_gcd = math.gcd(current, num)
                current = (current * num) // current_gcd
            return current
        
        max_score = 0
        
        # Compute original score
        original_gcd = compute_gcd(nums)
        original_lcm = compute_lcm(nums)
        original_score = original_gcd * original_lcm
        max_score = max(max_score, original_score)
        
        # Check each possible exclusion
        for i in range(len(nums)):
            new_list = nums[:i] + nums[i+1:]
            if not new_list:
                current_score = 0
            else:
                new_gcd = compute_gcd(new_list)
                new_lcm = compute_lcm(new_list)
                current_score = new_gcd * new_lcm
            if current_score > max_score:
                max_score = current_score
        
        return max_score