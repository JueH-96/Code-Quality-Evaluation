from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        if n == 0:
            return 0
        
        # Precompute sets for each property list
        sets = [set(prop) for prop in properties]
        
        parent = list(range(n))
        rank = [1] * n
        
        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # path compression
                u = parent[u]
            return u
        
        def union(u: int, v: int) -> None:
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return
            # Merge smaller rank into larger rank
            if rank[u_root] < rank[v_root]:
                parent[u_root] = v_root
            else:
                parent[v_root] = u_root
                if rank[u_root] == rank[v_root]:
                    rank[u_root] += 1
        
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if len(sets[i] & sets[j]) >= k:
                    union(i, j)
        
        # Count the number of unique roots
        roots = set()
        for i in range(n):
            roots.add(find(i))
        
        return len(roots)