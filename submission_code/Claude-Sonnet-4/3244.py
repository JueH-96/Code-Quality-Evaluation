class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        # Find GCD of all numbers
        overall_gcd = reduce(gcd, nums)
        
        # Count how many numbers equal the GCD
        gcd_count = nums.count(overall_gcd)
        
        # Each pair of GCD values can be reduced to 0
        # So we need ceiling division: (gcd_count + 1) // 2
        return (gcd_count + 1) // 2