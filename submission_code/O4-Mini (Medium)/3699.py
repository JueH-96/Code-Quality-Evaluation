from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        # Build packed list of (product, p, r) encoded in one integer:
        # prod uses bits >=20, p uses bits [10..19], r uses bits [0..9]
        keys = []
        for r in range(2, n):
            vr = nums[r]
            # p must satisfy r-p >= 2 => p <= r-2
            lim = r - 1
            for p in range(lim):
                prod = nums[p] * vr
                # pack into one integer
                key = (prod << 20) | (p << 10) | r
                keys.append(key)
        # sort by prod, then p, then r
        keys.sort()
        # Fenwick tree for counts of r-indices
        size = n
        bit = [0] * (size + 1)
        def ft_add(i: int, v: int) -> None:
            # add v at index i (0-based)
            i += 1
            while i <= size:
                bit[i] += v
                i += i & -i
        def ft_sum(i: int) -> int:
            # sum from 0..i inclusive
            if i < 0:
                return 0
            s = 0
            i += 1
            if i > size:
                i = size
            while i > 0:
                s += bit[i]
                i -= i & -i
            return s

        ans = 0
        prev_prod = -1
        # track which r's we have added, to reset after each product group
        updates = []
        for key in keys:
            prod = key >> 20
            if prod != prev_prod:
                # reset BIT for previous group
                for rr in updates:
                    ft_add(rr, -1)
                updates.clear()
                prev_prod = prod
            # unpack p, r
            p = (key >> 10) & 0x3FF
            r = key & 0x3FF
            # for this pair as second pair (q,r2) with q=p, r2=r,
            # count how many first-pairs had r1 <= p-2
            cnt = ft_sum(p - 2)
            ans += cnt
            # now add this pair as a possible first-pair for future
            ft_add(r, 1)
            updates.append(r)
        # no need final reset
        return ans