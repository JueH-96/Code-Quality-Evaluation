from typing import List

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        # Separate values <= threshold (they can connect) and > threshold (isolated)
        small = []
        for v in nums:
            if v <= threshold:
                small.append(v)
        large_count = n - len(small)
        
        # If no small values, all are isolated
        if not small:
            return n
        
        # Map each small value to a DSU index
        idx = {v: i for i, v in enumerate(small)}
        dsu = DSU(len(small))
        
        # Build buckets: for each m up to threshold, which small values divide m
        buckets = [[] for _ in range(threshold + 1)]
        for v in small:
            vid = idx[v]
            # mark all multiples of v
            for m in range(v, threshold + 1, v):
                buckets[m].append(vid)
        
        # For each bucket with ≥2 entries, union them all
        for m in range(1, threshold + 1):
            bucket = buckets[m]
            if len(bucket) > 1:
                first = bucket[0]
                for other in bucket[1:]:
                    dsu.union(first, other)
        
        # Count distinct roots among small
        roots = set()
        for i in range(len(small)):
            roots.add(dsu.find(i))
        small_components = len(roots)
        
        # Total components = components among small + each large as isolated
        return small_components + large_count