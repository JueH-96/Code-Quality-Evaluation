from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Pair up and sort by b ascending
        items = list(zip(nums1, nums2))
        items.sort(key=lambda p: p[1])  # sort by b_i
        
        # Compute sum of a_i and sum of b_i
        sum_a = sum(nums1)
        sum_b = sum(nums2)
        
        # dp[j] = maximum total deletion value D when selecting j items
        # dp state initialized for j=0..n
        # Use a large negative for unreachable states
        NEG_INF = -10**30
        dp = [NEG_INF] * (n + 1)
        dp[0] = 0
        
        # Build dp: for each item i in sorted order (b ascending),
        # if we pick it as the j-th in our selected set, it contributes a_i + b_i*j
        for a_i, b_i in items:
            # iterate j from current max down to 1
            # at most i items can be selected after processing i items,
            # but we just do j from n down to 1 safely
            for j in range(n, 0, -1):
                if dp[j-1] != NEG_INF:
                    # if we take this item as the j-th (i.e., it gets L=j)
                    val = dp[j-1] + a_i + b_i * j
                    if val > dp[j]:
                        dp[j] = val
        
        # Try times t from 0..n, compute final sum S(t) = sum(a_i + t*b_i) - D(t)
        # where D(t) = dp[t]. Return smallest t with S(t) <= x
        for t in range(n+1):
            # sum after increments before resets
            S0 = sum_a + t * sum_b
            D = dp[t]
            # if we can't reach this selection size, dp[t] is NEG_INF
            if D < 0:
                # if t>0 and dp[t] is negative, skip
                # but for t=0, dp[0]=0 always valid
                continue
            if S0 - D <= x:
                return t
        
        # if no t up to n works, impossible
        return -1