from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp1_masks[p] will hold a bitmask of all possible OR-values
        # for picking k elements from nums[0..p-1]
        dp1_masks = [0] * (n + 1)
        # dp1_cur[c] is a bitmask of OR-values achievable with c picks so far
        dp1_cur = [0] * (k + 1)
        dp1_cur[0] = 1  # only OR=0 with zero picks
        dp1_masks[0] = dp1_cur[k]  # at p=0, can't pick k>0 so it's 0
        
        # Build dp1 forward
        for i, v in enumerate(nums):
            # update counts from high to low to avoid reuse in same iteration
            top = min(i + 1, k)
            for c in range(top, 0, -1):
                prev_mask = dp1_cur[c - 1]
                if prev_mask:
                    m = prev_mask
                    newbits = 0
                    # for each bit u set in prev_mask, set bit (u|v)
                    while m:
                        lsb = m & -m
                        u = lsb.bit_length() - 1
                        m ^= lsb
                        newbits |= 1 << (u | v)
                    dp1_cur[c] |= newbits
            dp1_masks[i + 1] = dp1_cur[k]
        
        # dp2_masks[p] will hold a bitmask of all possible OR-values
        # for picking k elements from nums[p..n-1]
        dp2_masks = [0] * (n + 1)
        dp2_cur = [0] * (k + 1)
        dp2_cur[0] = 1
        dp2_masks[n] = dp2_cur[k]
        
        # Build dp2 backward
        for i in range(n - 1, -1, -1):
            v = nums[i]
            top = min(n - i, k)
            for c in range(top, 0, -1):
                prev_mask = dp2_cur[c - 1]
                if prev_mask:
                    m = prev_mask
                    newbits = 0
                    while m:
                        lsb = m & -m
                        u = lsb.bit_length() - 1
                        m ^= lsb
                        newbits |= 1 << (u | v)
                    dp2_cur[c] |= newbits
            dp2_masks[i] = dp2_cur[k]
        
        # Combine dp1 and dp2 at every valid split p
        ans = 0
        # p must allow k picks before and k picks after: k <= p <= n-k
        for p in range(k, n - k + 1):
            m1 = dp1_masks[p]
            m2 = dp2_masks[p]
            if not m1 or not m2:
                continue
            # extract OR-values u from m1
            u_vals = []
            m = m1
            while m:
                lsb = m & -m
                u_vals.append(lsb.bit_length() - 1)
                m ^= lsb
            # extract OR-values v from m2
            v_vals = []
            m = m2
            while m:
                lsb = m & -m
                v_vals.append(lsb.bit_length() - 1)
                m ^= lsb
            # brute-force their XORs
            for u in u_vals:
                for v in v_vals:
                    x = u ^ v
                    if x > ans:
                        ans = x
        return ans