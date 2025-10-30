from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        sets_list = [set(arr) for arr in properties]
        parent = list(range(n))
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        for i in range(n):
            for j in range(i + 1, n):
                if len(sets_list[i] & sets_list[j]) >= k:
                    root_i = find(i)
                    root_j = find(j)
                    if root_i != root_j:
                        parent[root_j] = root_i
        
        roots = set()
        for i in range(n):
            roots.add(find(i))
        return len(roots)