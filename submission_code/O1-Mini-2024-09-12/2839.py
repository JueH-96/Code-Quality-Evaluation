from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        class BIT:
            def __init__(self, size):
                self.N = size + 2
                self.tree = [float('-inf')] * (self.N)

            def update(self, idx, val):
                while idx < self.N:
                    if val > self.tree[idx]:
                        self.tree[idx] = val
                    else:
                        break
                    idx += idx & -idx

            def query(self, idx):
                res = float('-inf')
                while idx > 0:
                    if self.tree[idx] > res:
                        res = self.tree[idx]
                    idx -= idx & -idx
                return res

        n = len(nums1)
        m = len(queries)
        data = sorted(zip(nums1, nums2), key=lambda x: -x[0])
        sorted_queries = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], key=lambda x: -x[0])
        ys = sorted(list(set(nums2 + [y for _, y, _ in sorted_queries])))
        y_id = {y: i+1 for i, y in enumerate(ys)}
        bit = BIT(len(ys))
        res = [-1] * m
        ptr = 0
        for x, y, idx in sorted_queries:
            while ptr < n and data[ptr][0] >= x:
                _, num2 = data[ptr]
                bit.update(y_id[num2], data[ptr][0] + data[ptr][1])
                ptr += 1
            pos = bisect.bisect_left(ys, y)
            if pos < len(ys):
                q = bit.query(len(ys)) - bit.query(pos)
                res[idx] = bit.query(len(ys)) if bit.query(len(ys)) != float('-inf') else -1
            else:
                res[idx] = -1
        return res