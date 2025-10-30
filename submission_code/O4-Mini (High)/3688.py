from typing import List

# Range Minimum Query (segment tree) for prefix‐sum minima
class RMQ:
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        # next power of two
        sz = 1
        while sz < self.n:
            sz <<= 1
        self.size = sz
        INF = 10**30
        self.tree = [INF] * (2 * sz)
        # build leaves
        for i, v in enumerate(arr):
            self.tree[sz + i] = v
        # build internals
        for i in range(sz - 1, 0, -1):
            self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])

    # query min over arr[l..r] inclusive
    def query(self, l: int, r: int) -> int:
        INF = 10**30
        res = INF
        l += self.size
        r += self.size
        while l <= r:
            if (l & 1) == 1:
                if self.tree[l] < res:
                    res = self.tree[l]
                l += 1
            if (r & 1) == 0:
                if self.tree[r] < res:
                    res = self.tree[r]
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        # 1) Compute f_none = max subarray sum without any deletion
        cur = 0
        f_none = float('-inf')
        for v in nums:
            # standard Kadane
            cur = v if cur < 0 else cur + v
            if cur > f_none:
                f_none = cur

        # 2) Collect positions of each negative value (only negatives matter)
        pos_map = {}
        for i, v in enumerate(nums):
            if v < 0:
                pos_map.setdefault(v, []).append(i)
        # if no negative value, best is f_none
        if not pos_map:
            return f_none

        # 3) Build prefix sums P
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]
        rmq1 = RMQ(P)

        # 4) For each x<0 compute dp_end at its occurrences
        INF = 10**30
        fwd_vals = {}
        for x, poslist in pos_map.items():
            occ = len(poslist)
            fwd = [0] * occ
            best_B = INF
            prev_p = 0
            # j = number of occurrences before the j-th
            for j in range(occ):
                p = poslist[j] + 1
                # min of P[t] for t in [prev_p..p-1]
                mn = rmq1.query(prev_p, p-1)
                # adjust by -x * (count so far = j)
                seg_b = mn - x * j
                if seg_b < best_B:
                    best_B = seg_b
                # dp_end at this occurrence
                dp = P[p] - x * (j+1) - best_B
                fwd[j] = dp
                prev_p = p
            fwd_vals[x] = fwd

        # 5) Build reversed prefix sums RP
        rev = nums[::-1]
        RP = [0] * (n + 1)
        for i in range(n):
            RP[i+1] = RP[i] + rev[i]
        rmq2 = RMQ(RP)

        # 6) Combine forward and reverse dps to get best g(x)
        ans = f_none
        for x, poslist in pos_map.items():
            occ = len(poslist)
            # reversed positions in the reversed array
            # reversing poslist (descending) then mapping to rev‐indices yields sorted
            revpos = [ (n-1 - p) for p in reversed(poslist) ]
            best_B = INF
            prev_p = 0
            fwd = fwd_vals[x]
            # j_rev is count of occurrences before in rev‐array
            for j_rev, rp in enumerate(revpos):
                p = rp + 1
                mn = rmq2.query(prev_p, p-1)
                seg_b = mn - x * j_rev
                if seg_b < best_B:
                    best_B = seg_b
                dp_rev = RP[p] - x * (j_rev+1) - best_B
                # map back to forward index j
                j = occ - j_rev - 1
                total = fwd[j] + dp_rev
                if total > ans:
                    ans = total
                prev_p = p

        return ans