import math
from typing import List
from collections import defaultdict

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Count nodes with value greater than threshold, each is a separate component
        large_count = sum(1 for num in nums if num > threshold)
        
        # Filter nodes with value less than or equal to threshold
        small_nums = [num for num in nums if num <= threshold]
        if not small_nums:
            return large_count  # No small nodes, return count of large nodes
        
        # Create a set for fast lookup of small numbers
        val_set = set(small_nums)
        
        # Precompute divisors for each number up to threshold
        divs = [[] for _ in range(threshold + 1)]
        for d in range(1, threshold + 1):
            for multiple in range(d, threshold + 1, d):
                divs[multiple].append(d)
        
        # Initialize DSU
        parent = {num: num for num in small_nums}
        rank = {num: 0 for num in small_nums}
        
        # Define find and union functions
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px = find(x)
            py = find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
        
        # For each k from 1 to threshold, union all divisors in small_nums
        for k in range(1, threshold + 1):
            divisors_k = divs[k]
            div_in_s = [d for d in divisors_k if d in val_set]
            if len(div_in_s) > 1:
                # Union all to the first one
                rep = div_in_s[0]
                for other in div_in_s[1:]:
                    union(rep, other)
        
        # Find the number of components in small nodes
        roots = set()
        for num in small_nums:
            roots.add(find(num))  # Find applies path compression
        num_components_small = len(roots)
        
        # Total components
        return large_count + num_components_small