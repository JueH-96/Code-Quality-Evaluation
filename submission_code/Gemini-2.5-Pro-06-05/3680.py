import math
from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        
        class DSU:
            """A Disjoint Set Union (Union-Find) data structure."""
            def __init__(self, n: int):
                self.parent = list(range(n))

            def find(self, i: int) -> int:
                if self.parent[i] == i:
                    return i
                self.parent[i] = self.find(self.parent[i])
                return self.parent[i]

            def union(self, i: int, j: int) -> bool:
                root_i = self.find(i)
                root_j = self.find(j)
                if root_i != root_j:
                    # A simple union-by-rank/size heuristic could be used,
                    # but for this problem, it's not strictly necessary.
                    if root_i < root_j:
                        self.parent[root_j] = root_i
                    else:
                        self.parent[root_i] = root_j
                    return True
                return False

        # Any number in `nums` greater than `threshold` is isolated.
        # lcm(u, v) >= max(u, v), so if u > threshold, lcm(u, v) > threshold.
        isolated_count = sum(1 for x in nums if x > threshold)
        
        # We only need to consider numbers <= threshold.
        # Filter `nums` to get a set of relevant numbers for quick lookups.
        nums_in_range = {x for x in nums if x <= threshold}

        # DSU on numbers from 1 to threshold.
        dsu = DSU(threshold + 1)

        # The core idea is that any two numbers `u, v <= threshold` that share a common factor `g > 1`
        # can be shown to belong to the same connected component. This is because there exists
        # a path between them, potentially through other multiples of `g`.
        # We can thus group all numbers that are not mutually coprime.
        # We iterate through each potential common factor `g`.
        for g in range(2, threshold + 1):
            # Optimization: If we've processed a smaller prime factor of `g`,
            # its multiples are already handled. A sieve could optimize this further,
            # but this approach is sufficient.
            if dsu.find(g) != g:
                continue
            
            # Union all multiples of `g`.
            for multiple in range(2 * g, threshold + 1, g):
                dsu.union(g, multiple)
        
        # Count the number of distinct components for the numbers present in `nums_in_range`.
        components = set()
        for x in nums_in_range:
            # Find the root of the component `x` belongs to.
            root = dsu.find(x)
            components.add(root)
            
        num_components_in_range = len(components)
        
        return num_components_in_range + isolated_count