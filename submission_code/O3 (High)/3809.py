from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        if n == 0:
            return 0                       # just a guard, though n ≥ 1 per constraints

        # Convert each property list to a set of distinct integers
        sets = [set(row) for row in properties]

        # -------- Disjoint-set (Union–Find) helpers ----------
        parent = list(range(n))
        rank   = [0] * n                   # union by rank

        def find(x: int) -> int:
            while parent[x] != x:          # path-compression
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        # -----------------------------------------------------

        # Add an edge (i, j) if the two sets share ≥ k distinct integers
        for i in range(n - 1):
            for j in range(i + 1, n):
                if len(sets[i] & sets[j]) >= k:
                    union(i, j)

        # Count different roots → number of connected components
        roots = {find(i) for i in range(n)}
        return len(roots)