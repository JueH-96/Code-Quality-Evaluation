from math import gcd
from functools import reduce

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        current_gcd = nums[0]
        for num in nums[1:]:
            current_gcd = gcd(current_gcd, num)
            if current_gcd == 0:
                break  # Early exit if GCD becomes zero
        return 1 if current_gcd != 0 else len(nums)