class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        # Calculate the GCD of all elements in the array
        overall_gcd = reduce(gcd, nums)
        
        # If the GCD of all elements is greater than 1, the minimum length is 1
        # because we can reduce all elements to the GCD and then to zero except one
        return 1 if overall_gcd > 1 else len(nums)