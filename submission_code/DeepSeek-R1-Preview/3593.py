from typing import List
from math import gcd

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        max_score = 0
        candidates = [nums.copy()]  # Include the original array as a candidate
        for i in range(len(nums)):
            new_nums = nums[:i] + nums[i+1:]
            candidates.append(new_nums)
        
        for candidate in candidates:
            if not candidate:
                score = 0
            else:
                # Compute GCD
                current_gcd = candidate[0]
                for num in candidate[1:]:
                    current_gcd = gcd(current_gcd, num)
                    if current_gcd == 1:
                        break  # GCD can't be lower than 1
                # Compute LCM
                current_lcm = candidate[0]
                for num in candidate[1:]:
                    current_lcm = (current_lcm * num) // gcd(current_lcm, num)
                score = current_gcd * current_lcm
            if score > max_score:
                max_score = score
        
        return max_score