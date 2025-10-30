from typing import List, Tuple
import math

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # ------------------------------------------------------------
        # 1. next_greater[i]  : first index > nums[i] to its right
        # ------------------------------------------------------------
        next_greater = [n] * n
        st = []                       # monotone decreasing stack
        for i in range(n - 1, -1, -1):
            while st and nums[st[-1]] <= nums[i]:   # strictly greater
                st.pop()
            next_greater[i] = st[-1] if st else n
            st.append(i)

        # ------------------------------------------------------------
        # 2. binary–lifting tables
        #    jump[p][i]   : index reached after 2**p  “max-jumps”
        #    pref[p][i]   : contribution of those 2**p segments to the
        #                   sum_{j = i .. (jump[p][i]-1)}  prefix-max
        # ------------------------------------------------------------
        LOG = (n).bit_length()        # enough to cover n
        jump  = [[n] * n for _ in range(LOG)]
        pref  = [[0] * n for _ in range(LOG)]

        for i in range(n):
            j = next_greater[i]
            jump[0][i] = j
            pref[0][i] = nums[i] * (j - i)          # whole first segment

        for p in range(1, LOG):
            jp = jump[p - 1]
            pj = jump[p]
            ppref = pref[p]
            pprev = pref[p - 1]
            ppref_prev = pref[p - 1]
            for i in range(n):
                mid = jp[i]
                if mid < n:
                    jp_next = jump[p - 1][mid]
                    pj[i] = jp_next
                    ppref[i] = pprev[i] + ppref_prev[mid]
                else:
                    pj[i] = n
                    ppref[i] = pprev[i]

        # ------------------------------------------------------------
        # prefix sum of original numbers for fast range-sum queries
        # ------------------------------------------------------------
        ps = [0] * (n + 1)
        for i in range(n):
            ps[i + 1] = ps[i] + nums[i]

        # ------------------------------------------------------------
        # helper: sum_{t=l..r}  max_{l..t} nums
        # ------------------------------------------------------------
        def sum_prefix_max(l: int, r: int) -> int:
            res = 0
            idx = l
            while True:
                if idx > r:
                    break
                if next_greater[idx] > r:            # last (partial) segment
                    res += nums[idx] * (r - idx + 1)
                    break
                # use biggest power that keeps segment inside [idx, r]
                for p in range(LOG - 1, -1, -1):
                    if jump[p][idx] <= r:
                        res += pref[p][idx]
                        idx = jump[p][idx]
                        break
            return res

        # ------------------------------------------------------------
        # main: for every left boundary use binary search on right
        # ------------------------------------------------------------
        ans = 0
        best_r_prev = -1               # farthest valid right for previous l
        for l in range(n):
            low = max(l, best_r_prev)      # monotonicity
            hi = n - 1
            best = l - 1
            while low <= hi:
                mid = (low + hi) // 2
                need = sum_prefix_max(l, mid) - (ps[mid + 1] - ps[l])
                if need <= k:
                    best = mid
                    low = mid + 1
                else:
                    hi = mid - 1
            best_r_prev = best
            ans += best - l + 1
        return ans