from typing import List
import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Key observation:
        # You may “absorb” any number that is strictly larger than the minimum
        # by pairing it with an occurrence of the smallest number m.
        # Indeed, if m = min(nums) and x > m then choosing the pair (m, x)
        # in the order m % x gives m (since m < x, m % x = m).
        # (The operation removes both and then appends the remainder.
        # In our case that “saves” an m.)
        #
        # Thus, if at least one m exists (and there always is)
        # you can use it to “eliminate” every number that is not m,
        # leaving you with exactly count_m copies of m.
        # At that point every valid operation would need to use two m’s.
        # But note that m % m = 0.
        # So whenever you combine two copies of m you lose both positives and add one 0,
        # and since zeros (0) cannot be used in future operations, you cannot combine further.
        #
        # Therefore, starting with count_m copies of m you can pair them up as many
        # times as possible. In each pair the two m's become a single (dead) 0.
        # If count_m is even, pairing all m’s leaves count_m/2 elements (all zeros).
        # If count_m is odd, you can pair all but one – yielding (count_m-1)/2 zeros plus one m.
        #
        # In either case the total number (of survivors that can no longer be reduced)
        # is exactly math.ceil(count_m/2).
        
        m = min(nums)
        count_min = nums.count(m)
        return (count_min + 1) // 2  # equivalent to math.ceil(count_min/2)