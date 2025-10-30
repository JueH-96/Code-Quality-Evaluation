from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Build the frequency array f on the "final value" axis.
        # Every element a in nums produces a final value a (if unchanged) or a+1 (if increased)
        # The largest final value possible is max(nums) + 1.
        max_val = max(nums)
        M = max_val + 1  # final numbers range from 1 to M (inclusive)
        # We'll build an array f from index 0 to M (inclusive)
        # f[x] = number of elements that are originally equal to x.
        f = [0] * (M + 1)  # indices 0,...,M
        for a in nums:
            f[a] += 1
        
        # We will set up two DP arrays for indices 1 ... M+1 (we use M+1 as a base with 0 chain length).
        dp0 = [0] * (M + 2)  # dp[x][0]: if no carry is incoming at x
        dp1 = [0] * (M + 2)  # dp[x][1]: if carry is available at x
        # Base: for x = M+1, there are no targets left so dp values are 0.
        dp0[M+1] = 0
        dp1[M+1] = 0
        
        # Fill dp arrays downward for x = M, M-1, ... , 1.
        for x in range(M, 0, -1):
            # For dp1[x]: we have a carry available (the element came from x-1, increased to x)
            if f[x] > 0:
                # Since f(x)>0, we use a carry (do not "use" any fresh element) and then
                # we get all of f(x) available fresh (effectively a carry for next step).
                dp1[x] = 1 + dp1[x+1]
            else:
                # No fresh ones here. We use the carried element but then have no new reserve.
                dp1[x] = 1 + dp0[x+1]
            
            # For dp0[x]: no carry is available so we must use a fresh element from those equal to x.
            if f[x] >= 1:
                if f[x] > 1:
                    # After using one from f(x), at least one is left to “carry over”
                    dp0[x] = 1 + dp1[x+1]
                else:  # f[x] is exactly 1, so after using it, nothing is left to act as a carry.
                    dp0[x] = 1 + dp0[x+1]
            else:
                dp0[x] = 0  # cannot cover x if we have no fresh candidate
        
        # Now, we consider the choice of starting final value L.
        # When we start at L, note that an element originally equal to (L-1) (if any) could be increased so that
        # the final value is L. Hence the initial state is: flag = 1 if f[L-1] > 0, else 0.
        ans = 0
        # We only consider L from 1 to M (M = max(nums)+1) because these are possible final numbers.
        for L in range(1, M+1):
            if L - 1 >= 0 and f[L-1] > 0:
                candidate = dp1[L]
            else:
                candidate = dp0[L]
            if candidate > ans:
                ans = candidate
        return ans