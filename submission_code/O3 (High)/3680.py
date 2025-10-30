from typing import List

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return
        if self.rank[pa] < self.rank[pb]:
            self.parent[pa] = pb
        elif self.rank[pb] < self.rank[pa]:
            self.parent[pb] = pa
        else:
            self.parent[pb] = pa
            self.rank[pa] += 1


class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        # Map value -> its index in the original array
        idx_of_val = {val: i for i, val in enumerate(nums)}

        dsu = DSU(n)

        # helper array: first_idx[m] is the index of the first number that divides m
        first_idx = [-1] * (threshold + 1)

        for val in nums:
            if val > threshold:          # numbers larger than threshold are isolated
                continue
            cur_idx = idx_of_val[val]
            # iterate through all multiples of val up to threshold
            for m in range(val, threshold + 1, val):
                if first_idx[m] == -1:
                    first_idx[m] = cur_idx
                else:
                    dsu.union(cur_idx, first_idx[m])

        # count distinct parents
        roots = set(dsu.find(i) for i in range(n))
        return len(roots)