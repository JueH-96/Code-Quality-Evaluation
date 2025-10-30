import math
from bisect import bisect_right

class Solution:
    def countComponents(self, nums: list[int], threshold: int) -> int:
        # Separate the numbers into those <= threshold and > threshold
        S = [x for x in nums if x <= threshold]
        L = [x for x in nums if x > threshold]
        
        if not S:
            return len(L)
        
        # Sort the list S for efficient processing
        S_sorted = sorted(S)
        s_set = set(S)  # For quick existence checks
        
        # Initialize the Union-Find structure
        parent = {}
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        
        # Initialize each node as its own parent
        for x in S:
            parent[x] = x
        
        # Connect each number to its multiples in S
        for x in S_sorted:
            m = 2
            while True:
                y = x * m
                if y > threshold:
                    break
                if y in s_set:
                    if y not in parent:
                        parent[y] = y
                    union(x, y)
                m += 1
        
        # Connect each number x to all y where y > x and LCM(x, y) <= threshold
        n = len(S_sorted)
        for i in range(n):
            x = S_sorted[i]
            max_y = threshold // x
            # Find the rightmost index where y <= max_y
            j = bisect_right(S_sorted, max_y, i + 1, n)
            for k in range(i + 1, j):
                y = S_sorted[k]
                gcd = math.gcd(x, y)
                lcm = (x * y) // gcd
                if lcm <= threshold:
                    union(x, y)
        
        # Count the number of unique components in S
        components = set()
        for x in S:
            components.add(find(x))
        
        # The total number of components is the components in S plus those in L
        return len(components) + len(L)