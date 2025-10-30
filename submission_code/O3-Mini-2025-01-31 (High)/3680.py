from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        # Union-Find data structure with path compression and union by rank.
        parent = list(range(n))
        rank = [0] * n
        
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x: int, y: int):
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
        
        # We'll connect only those numbers <= threshold.
        # Nodes with nums[i] > threshold are isolated.
        # For numbers in group A (nums[i] <= threshold): Two nodes (with values a and b)
        # are connected if there exists a common multiple d (with 1 <= d <= threshold)
        # such that a divides d and b divides d. Note that this is equivalent to lcm(a, b) <= threshold.
        #
        # To efficiently “simulate” this connection without testing every pair,
        # we build an array 'divs' where for each candidate d (1 <= d <= threshold)
        # we collect all nodes i in group A whose value divides d.
        # Then, for each d for which we get at least 2 nodes, we union those nodes.
        
        divs = [[] for _ in range(threshold + 1)]
        
        # Process each number that is <= threshold (group A).
        for i, x in enumerate(nums):
            if x <= threshold:
                # For each d that is a multiple of x (i.e. x divides d),
                # add the node i. This d is a candidate common multiple.
                for d in range(x, threshold + 1, x):
                    divs[d].append(i)
            # Numbers with x > threshold are left untouched (group B, isolated nodes).
        
        # For each candidate d in 1..threshold, connect all nodes that divide d.
        for d in range(1, threshold + 1):
            if len(divs[d]) > 1:
                base = divs[d][0]
                for j in range(1, len(divs[d])):
                    union(base, divs[d][j])
        
        # Count unique components.
        # For group A (nums[i] <= threshold) we count unique union-find leaders,
        # while every group B node (nums[i] > threshold) forms its own component.
        seen = set()
        for i, x in enumerate(nums):
            if x <= threshold:
                seen.add(find(i))
        countA = len(seen)
        countB = sum(1 for x in nums if x > threshold)
        return countA + countB