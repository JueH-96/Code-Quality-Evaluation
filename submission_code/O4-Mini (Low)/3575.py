from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp1_cur[t]: bitmask of reachable OR-values after picking t items from the current prefix
        # dp2_cur[t]: bitmask for suffix
        # We only need to record dp1_res[i] = dp1_cur[k] after processing up to i,
        # and dp2_res[i] = dp2_cur[k] after processing from i to end.
        
        # Build prefix DP
        dp1_cur = [0] * (k + 1)
        dp1_cur[0] = 1 << 0  # OR=0 is possible with 0 picks
        dp1_res = [0] * n   # dp1_res[i] will hold bitmask of OR-values for t=k using nums[:i+1]
        
        for i, v in enumerate(nums):
            # update dp1_cur in reverse t order to avoid overwriting
            for t in range(min(k, i + 1), 0, -1):
                prev_mask = dp1_cur[t - 1]
                new_mask = 0
                # for each OR-value 'a' that was possible with t-1 picks,
                # we can form a|v with one more pick
                m = prev_mask
                while m:
                    a = (m & -m).bit_length() - 1  # extract lowest set bit
                    m &= m - 1
                    new_mask |= 1 << (a | v)
                dp1_cur[t] |= new_mask
            # record for exactly k picks
            dp1_res[i] = dp1_cur[k]
        
        # Build suffix DP
        dp2_cur = [0] * (k + 1)
        dp2_cur[0] = 1 << 0
        dp2_res = [0] * n  # dp2_res[i] will hold bitmask for nums[i:]
        
        for i in range(n - 1, -1, -1):
            v = nums[i]
            for t in range(min(k, n - i), 0, -1):
                prev_mask = dp2_cur[t - 1]
                new_mask = 0
                m = prev_mask
                while m:
                    a = (m & -m).bit_length() - 1
                    m &= m - 1
                    new_mask |= 1 << (a | v)
                dp2_cur[t] |= new_mask
            dp2_res[i] = dp2_cur[k]
        
        # Now consider split positions i where prefix is nums[:i+1], suffix nums[i+1:]
        # We need at least k picks in each, so i from k-1 to n-k-1
        ans = 0
        for i in range(k - 1, n - k):
            mask_a = dp1_res[i]
            mask_b = dp2_res[i + 1]
            if mask_a == 0 or mask_b == 0:
                continue
            # extract all possible a's and b's
            a_vals = []
            m = mask_a
            while m:
                a = (m & -m).bit_length() - 1
                m &= m - 1
                a_vals.append(a)
            b_vals = []
            m = mask_b
            while m:
                b = (m & -m).bit_length() - 1
                m &= m - 1
                b_vals.append(b)
            # brute-force combine
            for a in a_vals:
                for b in b_vals:
                    x = a ^ b
                    if x > ans:
                        ans = x
        return ans