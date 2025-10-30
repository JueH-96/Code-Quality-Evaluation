from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Group positions by value
        groups = {}
        for i, v in enumerate(nums):
            groups.setdefault(v, []).append(i)
        # Process values in descending order
        items = sorted(groups.items(), key=lambda x: -x[0])

        # Fenwick tree (BIT) to mark positions of already-processed larger elements
        class BIT:
            def __init__(self, size: int):
                self.n = size
                self.ft = [0] * (size + 1)
            def add(self, i: int, val: int):
                while i <= self.n:
                    self.ft[i] += val
                    i += i & -i
            def sum(self, i: int) -> int:
                s = 0
                while i > 0:
                    s += self.ft[i]
                    i -= i & -i
                return s

        bit = BIT(n)
        pairs = 0

        # For each value, count pairs of occurrences not separated by a larger element
        for v, poslist in items:
            if len(poslist) > 1:
                cnt = 1
                prev = poslist[0]
                for p in poslist[1:]:
                    # Check if there is any larger element between prev and p
                    # bit.sum(p) sums markers at [0..p-1], bit.sum(prev+1) sums [0..prev]
                    if bit.sum(p) - bit.sum(prev + 1) == 0:
                        cnt += 1
                    else:
                        pairs += cnt * (cnt - 1) // 2
                        cnt = 1
                    prev = p
                pairs += cnt * (cnt - 1) // 2
            # Mark these positions as separators for smaller values
            for p in poslist:
                bit.add(p + 1, 1)

        # Add the n single-element subarrays
        return n + pairs