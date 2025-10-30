from typing import List
import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Calculate the GCD of all elements in the array
        gcd_value = nums[0]
        for num in nums[1:]:
            gcd_value = math.gcd(gcd_value, num)

        # If the GCD is 1, the minimum length is 1
        if gcd_value == 1:
            return 1
        # Otherwise, the minimum length is 2
        else:
            return 2