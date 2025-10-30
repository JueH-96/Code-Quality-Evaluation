from typing import List

class DSU:
    def __init__(self, elements):
        self.parent = {}
        for num in elements:
            self.parent[num] = num

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        S = set()
        s_count = 0
        for num in nums:
            if num > threshold:
                s_count += 1
            else:
                S.add(num)
        if not S:
            return s_count
        
        max_k = threshold
        # Precompute divisors for each k in 1..max_k
        divisors = [[] for _ in range(max_k + 1)]
        for d in range(1, max_k + 1):
            multiple = d
            while multiple <= max_k:
                divisors[multiple].append(d)
                multiple += d
        
        dsu = DSU(S)
        # Process each k from 1 to max_k
        for k in range(1, max_k + 1):
            group = []
            for d in divisors[k]:
                if d in S:
                    group.append(d)
            if len(group) < 1:
                continue
            first = group[0]
            for d in group[1:]:
                dsu.union(first, d)
        
        # Count unique roots
        roots = set()
        for num in S:
            roots.add(dsu.find(num))
        return len(roots) + s_count