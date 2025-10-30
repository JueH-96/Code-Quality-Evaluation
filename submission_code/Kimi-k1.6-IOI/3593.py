import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        max_score = 0
        candidates = [nums]
        # Generate all subarrays by removing one element
        for i in range(len(nums)):
            candidates.append(nums[:i] + nums[i+1:])
        
        for arr in candidates:
            if not arr:
                current_score = 0
            else:
                # Calculate GCD of the array
                gcd_val = arr[0]
                for num in arr[1:]:
                    gcd_val = math.gcd(gcd_val, num)
                    if gcd_val == 1:
                        break  # Early exit if GCD is 1
                
                # Calculate LCM of the array
                lcm_val = arr[0]
                for num in arr[1:]:
                    current_gcd = math.gcd(lcm_val, num)
                    lcm_val = (lcm_val * num) // current_gcd
                
                current_score = gcd_val * lcm_val
            
            if current_score > max_score:
                max_score = current_score
        
        return max_score