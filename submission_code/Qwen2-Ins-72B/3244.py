from math import gcd
from functools import reduce

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Find the GCD of all numbers in the list
        overall_gcd = reduce(gcd, nums)
        
        # Find the smallest number in the list that is divisible by the overall GCD
        smallest_divisible = min(num for num in nums if num % overall_gcd == 0)
        
        # The minimum length of the array is the smallest number divisible by the GCD divided by the GCD
        return smallest_divisible // overall_gcd