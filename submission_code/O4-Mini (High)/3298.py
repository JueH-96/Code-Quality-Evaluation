from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Edge case
        if not nums:
            return 0
        
        # Build a small-count array C such that C[v] = min(original_count[v], 2)
        maxv = max(nums)
        # We need indices 0..maxv+1 inclusive
        C = bytearray(maxv + 2)
        for v in nums:
            if C[v] < 2:
                C[v] += 1
        
        # dp0_next  = dp0[pos+1], dp1_next = dp1[pos+1]
        dp0_next = 0
        dp1_next = 0
        ans = 0
        
        # We sweep from high to low:
        # dp0[pos] = max chain length starting at pos with carry c=0
        # dp1[pos] = max chain length starting at pos with carry c=1
        #
        # Transitions:
        # if C[pos] + c == 0: dp = 0  (can't cover pos)
        # else
        #   c' = 1 if C[pos] + c >= 2 else 0
        #   dp = 1 + dp_{c'}[pos+1]
        #
        # We only count dp0[pos] if we actually have an original at pos (C[pos]>=1).
        # We count dp1[pos] as a valid start only if we can use an increment
        # from pos-1, i.e. C[pos-1]>=1.
        
        for pos in range(maxv + 1, -1, -1):
            cval = C[pos]
            
            # dp0 (start with carry=0)
            if cval >= 1:
                # we can cover pos by using an original
                if cval >= 2:
                    # leftover >=1 → carry into next
                    dp0_curr = 1 + dp1_next
                else:
                    dp0_curr = 1 + dp0_next
                # update answer for a start at pos using original
                if dp0_curr > ans:
                    ans = dp0_curr
            else:
                dp0_curr = 0
            
            # dp1 (start with carry=1)
            # we always have C[pos]+1 >= 1, so we always cover pos
            if cval >= 1:
                # C[pos]+1 >=2 → carry into next
                dp1_curr = 1 + dp1_next
            else:
                # C[pos] == 0 → C[pos]+1 == 1 → no carry
                dp1_curr = 1 + dp0_next
            
            # update answer for a start at pos using an increment
            # only if pos>0 and there was at least one original at pos-1
            if pos > 0 and C[pos-1] >= 1 and dp1_curr > ans:
                ans = dp1_curr
            
            # shift for next iteration
            dp0_next = dp0_curr
            dp1_next = dp1_curr
        
        return ans