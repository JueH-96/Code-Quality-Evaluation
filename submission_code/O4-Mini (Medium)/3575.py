from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp1_k[i] will be the set of possible OR values by choosing k elements
        # from nums[0..i]
        dp1_k = [set() for _ in range(n)]
        # dp1cur[j] = set of OR-values by choosing j elements so far in prefix
        dp1cur = [set() for _ in range(k+1)]
        dp1cur[0].add(0)
        for i, v in enumerate(nums):
            # update dp1cur in reverse j order
            for j in range(k, 0, -1):
                prev = dp1cur[j-1]
                if prev:
                    # OR v into each previous OR-value
                    # and add to dp1cur[j]
                    ors = {o | v for o in prev}
                    dp1cur[j].update(ors)
            # record dp1cur[k] for this position
            dp1_k[i] = set(dp1cur[k])
        
        # dp2_k[i] will be the set of possible OR values by choosing k elements
        # from nums[i..n-1]
        dp2_k = [set() for _ in range(n)]
        dp2cur = [set() for _ in range(k+1)]
        dp2cur[0].add(0)
        # build from the back
        for idx in range(n-1, -1, -1):
            v = nums[idx]
            for j in range(k, 0, -1):
                prev = dp2cur[j-1]
                if prev:
                    ors = {o | v for o in prev}
                    dp2cur[j].update(ors)
            dp2_k[idx] = set(dp2cur[k])
        
        # combine: split between pos and pos+1,
        # take k from prefix ending at pos, k from suffix starting at pos+1
        ans = 0
        # valid split positions: pos from 0..n-2,
        # but need at least k picks in prefix and k in suffix
        # so pos >= k-1 and pos+1 <= n-k => pos <= n-k-1
        lo = k-1
        hi = n - k - 1
        if lo <= hi:
            for pos in range(lo, hi+1):
                setA = dp1_k[pos]
                setB = dp2_k[pos+1]
                if not setA or not setB:
                    continue
                # try all combinations of OR_A XOR OR_B
                for a in setA:
                    # small optimization: if a XOR b max possible is when b is ~a
                    # but we just brute since sets are <=128
                    for b in setB:
                        val = a ^ b
                        if val > ans:
                            ans = val
        return ans