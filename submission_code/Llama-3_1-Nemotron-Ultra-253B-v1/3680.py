from typing import List
import bisect

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        small = [x for x in nums if x <= threshold]
        large = [x for x in nums if x > threshold]
        n = len(small)
        if n == 0:
            return len(large)
        
        small.sort()
        parent = {x: x for x in small}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            fx = find(x)
            fy = find(y)
            if fx != fy:
                parent[fy] = fx
        
        small_set = set(small)
        
        for i, x in enumerate(small):
            # Connect to multiples of x
            multiple = 2 * x
            while multiple <= threshold:
                if multiple in small_set:
                    union(x, multiple)
                multiple += x
            
            # Connect to y >= x and y <= threshold // x
            max_y = threshold // x
            j = bisect.bisect_right(small, max_y)
            for y in small[i+1:j]:
                union(x, y)
        
        roots = set()
        for x in small:
            roots.add(find(x))
        return len(roots) + len(large)