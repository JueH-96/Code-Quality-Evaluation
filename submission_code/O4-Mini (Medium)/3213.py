from typing import List

class BIT:
    def __init__(self, n):
        # 1-based BIT of size n (indices 1..n)
        self.n = n
        self.tree = [0] * (n+1)
        # largest power of two <= n
        self.max_log = 1
        while (1 << self.max_log) <= n:
            self.max_log += 1

    def add(self, i, delta):
        # add delta at index i
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def sum(self, i):
        # sum from 1 to i
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def find_kth(self, k):
        # find smallest i such that sum(i) >= k
        idx = 0
        bit_mask = 1 << (self.max_log - 1)
        while bit_mask > 0:
            nxt = idx + bit_mask
            if nxt <= self.n and self.tree[nxt] < k:
                k -= self.tree[nxt]
                idx = nxt
            bit_mask >>= 1
        return idx + 1

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Map each value to list of positions
        pos = {}
        for i, v in enumerate(nums):
            pos.setdefault(v, []).append(i)
        # Unique values sorted descending
        vals = sorted(pos.keys(), reverse=True)
        # BIT over positions -1..n mapped to indices 1..n+2 via p+2
        size = n + 2  # max index in BIT
        bit = BIT(size)
        # Insert sentinels at p=-1 -> idx=1, p=n -> idx=n+2
        bit.add(1, 1)
        bit.add(n+2, 1)
        total_seps = 2  # count of separators in BIT
        ans = 0
        for v in vals:
            plist = pos[v]
            # map to BIT indices
            pbit = [p + 2 for p in plist]
            t = len(pbit)
            if t >= k:
                # for each window of k occurrences
                for i in range(t - k + 1):
                    pi = pbit[i]
                    pj = pbit[i + k - 1]
                    # find separator to left of pi
                    cnt_left = bit.sum(pi - 1)
                    L_sep = bit.find_kth(cnt_left)
                    # prev M occurrence
                    if i > 0:
                        prev_M = pbit[i-1]
                    else:
                        prev_M = L_sep
                    p_prev = prev_M
                    # find separator to right of pj
                    sum_pj = bit.sum(pj)
                    cnt_total = total_seps
                    # next separator is the (sum_pj+1)-th in BIT
                    R_sep = bit.find_kth(sum_pj + 1)
                    # next M occurrence
                    if i + k < t:
                        p_next = pbit[i + k]
                    else:
                        p_next = R_sep
                    # compute choices
                    left_choices = pi - p_prev
                    right_choices = p_next - pj
                    ans += left_choices * right_choices
            # add these positions as separators for smaller values
            for pb in pbit:
                bit.add(pb, 1)
            total_seps += t
        return ans