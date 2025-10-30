from typing import List
from collections import Counter

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Build frequency table for available numbers.
        # Note: each element a is available as "a" (used directly) and can also be used 
        # to cover "a+1" (if not used in its own place). 
        # We only need to consider numbers from 1 to max(nums)+1.
        if not nums:
            return 0
        max_a = max(nums)
        M = max_a + 1  # maximum candidate starting value (and also block targets up to M+?)
        
        # Create frequency array for indices 1..M+1; index 0 unused.
        avail = [0] * (M + 2)
        for a in nums:
            if a < len(avail):
                avail[a] += 1
            else:
                # Should not happen because a <= max_a
                avail[a] += 1

        # Helper: check if block starting at x with length k is feasible.
        # That is, can we assign distinct elements to cover each integer v in
        # the block [x, x+k-1] according to the rule described.
        def can_cover(x: int, k: int) -> bool:
            # carry: available elements from the previous number that were not used,
            # and which can be increased to cover the current target.
            # For v = x, the candidates from "previous" are avail[x-1].
            # (If x-1 == 0, avail[0] is 0 by our construction.)
            carry = 0
            if x - 1 >= 0:
                carry = avail[x - 1]
            # Try to cover v = x, x+1, ..., x+k-1 in order.
            for v in range(x, x + k):
                # total available to cover v: available directly (avail[v])
                # plus the carry (from previous number v-1, which can only be used for v).
                total = carry + avail[v]
                if total < 1:
                    return False  # cannot cover v
                # We want to use as little as possible from the "carry".
                # Ideally, use elements from avail[v] first.
                # How many do we need to cover v? It's 1.
                use_direct = 1 if avail[v] >= 1 else 0
                needed_from_carry = 1 - use_direct
                # Use needed_from_carry from carry.
                carry -= needed_from_carry
                # Now, the leftover from avail[v] (which was not used directly)
                # can be carried over to cover future v+1.
                extra = avail[v] - use_direct
                carry += extra
            return True
        
        # We want the maximum chain length k for which there exists at least one starting x (1<=x<=M)
        # such that the block [x, x+k-1] is coverable.
        # Notice k cannot exceed the number of elements in nums.
        n = len(nums)
        
        # Binary search on k
        lo, hi = 1, n  # lower and upper bound for answer
        ans = 1  # at least one element can always be selected
        while lo <= hi:
            mid = (lo + hi) // 2
            found = False
            # For each candidate starting x from 1 to M,
            # note that the maximum number in a block starting at x is x+mid-1.
            # We only need to try those x such that x+mid-1 <= M+1 maybe.
            # However, since avail is defined up to M+1,
            # we simply try x from 1 to M+1-mid (since if x is too high, the block's maximum might be > M+1).
            for x in range(1, M + 2 - mid):
                if can_cover(x, mid):
                    found = True
                    break
            if found:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans