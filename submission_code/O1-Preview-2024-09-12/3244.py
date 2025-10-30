class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        import math
        from functools import reduce
        gcd = reduce(math.gcd, nums)
        if gcd == 1:
            return 1
        else:
            return 2