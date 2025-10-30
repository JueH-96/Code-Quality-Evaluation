from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Separate nums into S and T
        S = [x for x in nums if x <= threshold]
        T = [x for x in nums if x > threshold]
        
        if not S:
            return len(T)
        
        # Map each x in S to a unique index
        x_to_idx = {x: idx for idx, x in enumerate(S)}
        
        # Initialize Union-Find
        parent = list(range(len(S)))
        rank = [1] * len(S)
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1
        
        # Create a list of lists for each z from 1 to threshold
        divisor_lists = [[] for _ in range(threshold + 1)]
        for x in S:
            for m in range(x, threshold + 1, x):
                divisor_lists[m].append(x)
        
        # Union elements that divide the same z
        for z in range(1, threshold + 1):
            divisors = divisor_lists[z]
            if len(divisors) > 1:
                # Use the first element as the root
                root = x_to_idx[divisors[0]]
                for i in range(1, len(divisors)):
                    union(root, x_to_idx[divisors[i]])
        
        # Count unique roots
        unique_roots = set(find(x_to_idx[x]) for x in S)
        # Total components is the number of unique roots plus the size of T
        return len(unique_roots) + len(T)