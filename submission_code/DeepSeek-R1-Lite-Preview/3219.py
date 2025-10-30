from typing import List

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        uf = UnionFind(n)
        
        # Create list of (value, index)
        indexed_nums = [(nums[i], i) for i in range(n)]
        # Sort based on values
        indexed_nums.sort(key=lambda x: x[0])
        
        # Union indices where difference <= limit
        for i in range(1, n):
            if indexed_nums[i][0] - indexed_nums[i-1][0] <= limit:
                uf.union(indexed_nums[i][1], indexed_nums[i-1][1])
        
        # Group indices by their root parent
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            root = uf.find(i)
            groups[root].append(i)
        
        # For each group, sort the values and assign to sorted indices
        res = [0] * n
        for indices in groups.values():
            # Collect the values of the group
            group_values = [nums[i] for i in indices]
            # Sort the values
            group_values.sort()
            # Sort the indices
            indices_sorted = sorted(indices)
            # Assign the smallest values to the smallest indices
            for i, val in zip(indices_sorted, group_values):
                res[i] = val
        
        return res