from typing import List

class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        if n <= 1:
            return nums[:]  # trivial

        # The idea:
        # We can swap any two numbers if the difference in their values is <= limit.
        # This operation is transitive in the sense that if we can swap a<->b and b<->c,
        # then a, b, and c can be arbitrarily rearranged.
        # Our task is to find connected components of indices where there exists a chain
        # i1, i2, ..., ik with |nums[i_j]-nums[i_{j+1]]| <= limit.
        #
        # How to build these connectivity components efficiently?
        # Note that if we sort the indices by their corresponding nums value,
        # then any two indices that are adjacent in this sorted order and have difference <= limit 
        # will be connected. And by transitivity, a chain of such adjacencies produces a connected component.
        #
        # Thus we sort the indices by their nums value,
        # then union adjacent indices (in sorted order) if the difference in their values is <= limit.
        
        indexed = [(num, idx) for idx, num in enumerate(nums)]
        indexed.sort()  # sort by num value
        uf = UF(n)
        
        # union adjacent indices if gap <= limit
        for i in range(len(indexed) - 1):
            val1, idx1 = indexed[i]
            val2, idx2 = indexed[i+1]
            if val2 - val1 <= limit:
                uf.union(idx1, idx2)
        
        # Group indices by their root parent.
        groups = {}
        for i in range(n):
            root = uf.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # For each group, we can rearrange the values arbitrarily.
        # To obtain the lexicographically smallest overall array, 
        # for each connected component we sort the indices (so that we know the order in the final array)
        # and we also sort the list of corresponding values (so that the smallest values are placed in the smallest indices)
        result = list(nums)  # copy original list
        
        for group in groups.values():
            group.sort() 
            values = [nums[i] for i in group]
            values.sort()
            # assign sorted values to sorted indices
            for pos, val in zip(group, values):
                result[pos] = val
        
        return result