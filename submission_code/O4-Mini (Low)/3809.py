from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        # Disjoint Set Union (Union-Find) structure
        n = len(properties)
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
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
        
        # Convert each property list to a set for fast intersection
        prop_sets = [set(p) for p in properties]
        
        # For each pair i, j, if their intersection size >= k, union them
        for i in range(n):
            for j in range(i + 1, n):
                if len(prop_sets[i].intersection(prop_sets[j])) >= k:
                    union(i, j)
        
        # Count distinct roots
        roots = set(find(i) for i in range(n))
        return len(roots)

# Example usage:
# sol = Solution()
# print(sol.numberOfComponents([[1,2],[1,1],[3,4],[4,5],[5,6],[7,7]], 1))  # Output: 3
# print(sol.numberOfComponents([[1,2,3],[2,3,4],[4,3,5]], 2))              # Output: 1
# print(sol.numberOfComponents([[1,1],[1,1]], 2))                         # Output: 2