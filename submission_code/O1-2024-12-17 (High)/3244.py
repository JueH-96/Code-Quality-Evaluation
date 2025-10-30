class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        import math
        from math import gcd
        
        # 1) Compute the GCD of all elements.
        g = 0
        for x in nums:
            g = gcd(g, x)
        
        # If the overall gcd is 1, we can reduce everything down to 1 (and then no further),
        # so the final array length is 1.
        if g == 1:
            return 1
        
        # 2) Otherwise, gcd > 1.  We next check whether "every pair is a divisor pair."
        #    Concretely, after sorting, we should have a chain:
        #      a1 divides a2, a2 divides a3, ..., a(n-1) divides a(n)
        #    If this chain property fails for any consecutive pair, we can still
        #    combine remainders to eventually end up with one number, so answer = 1.
        nums.sort()
        chain = True
        for i in range(len(nums) - 1):
            if nums[i+1] % nums[i] != 0:
                chain = False
                break
        
        if not chain:
            # There's at least one pair that is not "divisor-related."
            # This allows us to perform enough mod operations (in a gcd-like chain)
            # so that we can end with a single number.
            return 1
        
        # 3) If we do have a "divisor chain," then eventually we end up with multiple
        #    copies of the smallest element (nums[0]).  Call that smallest value min_val,
        #    and let f be how many times it appears initially.  All larger elements
        #    can be removed one by one (paired with min_val) without changing the count
        #    of min_val.  In the end, we have f copies of min_val.  Pairing them among
        #    themselves yields zero remainders until we're left with either 0 or 1 copy
        #    (depending on parity).  The minimal final length = ceil(f / 2).
        min_val = nums[0]
        f = sum(1 for x in nums if x == min_val)
        
        return (f + 1) // 2