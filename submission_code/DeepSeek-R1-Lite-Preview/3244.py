from typing import List
import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Find the GCD of the array
        g = nums[0]
        for num in nums[1:]:
            g = math.gcd(g, num)
        
        # Count how many elements are equal to the GCD
        count_g = sum(1 for num in nums if num == g)
        
        if count_g == 0:
            return 1
        else:
            return (count_g + 1) // 2