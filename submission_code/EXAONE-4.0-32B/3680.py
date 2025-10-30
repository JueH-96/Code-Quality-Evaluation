import math
from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        setB = [x for x in nums if x > threshold]
        setA = [x for x in nums if x <= threshold]
        n_components = len(setB)
        
        if not setA:
            return n_components
        
        max_val = threshold
        exists = [False] * (max_val + 1)
        for x in setA:
            exists[x] = True
            
        parent = {}
        rank = {}
        for x in setA:
            parent[x] = x
            rank[x] = 0
            
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
                
        for d in range(1, threshold + 1):
            min_val = None
            x = d
            while x <= threshold:
                if exists[x]:
                    if min_val is None:
                        min_val = x
                    else:
                        g = math.gcd(min_val, x)
                        lcm_val = min_val * x // g
                        if lcm_val <= threshold:
                            union(min_val, x)
                x += d
                
        roots = set()
        for x in setA:
            roots.add(find(x))
        n_components += len(roots)
        return n_components