class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        """
        We want to reorder the array into the lexicographically smallest arrangement
        by swapping any two elements whose absolute difference in value does not exceed limit.
        
        Observing that if |nums[i] - nums[j]| <= limit, we can 'connect' i and j in a graph,
        then all indices in a connected component can freely swap values among themselves.
        Hence, within each connected component, we can sort both the indices and the values
        and assign the smallest values to the smallest indices for a lexicographically
        minimal solution.

        Steps:
        1) Pair each value with its index -> (value, index). Sort by value.
        2) Create a Disjoint Set (Union-Find) structure over indices [0..n-1].
        3) Traverse the sorted (value, index) array. Whenever consecutive pairs differ
           by at most limit, unite their indices in the Disjoint Set.
        4) Each connected component in the Disjoint Set thus collects certain indices.
           For those indices, gather their original values, sort them, sort the indices,
           then reassign the values in ascending order of both.
        5) Return the modified array.
        """

        # Disjoint Set (Union-Find) implementation
        class DS:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0]*n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx != ry:
                    if self.rank[rx] < self.rank[ry]:
                        rx, ry = ry, rx
                    self.parent[ry] = rx
                    if self.rank[rx] == self.rank[ry]:
                        self.rank[rx] += 1
        
        n = len(nums)
        if n <= 1:
            return nums

        # Sort by value
        val_idx = sorted((val, i) for i, val in enumerate(nums))
        
        # Build Disjoint Set over indices
        ds = DS(n)
        for i in range(n-1):
            val1, idx1 = val_idx[i]
            val2, idx2 = val_idx[i+1]
            if val2 - val1 <= limit:
                ds.union(idx1, idx2)
        
        from collections import defaultdict
        # Gather indices by connected component (root)
        comp = defaultdict(list)
        for i in range(n):
            root = ds.find(i)
            comp[root].append(i)
        
        # For each connected component, sort indices and sort the values
        # then assign smallest values to smallest indices
        ans = list(nums)
        for root, indices in comp.items():
            indices.sort()
            # Collect the values from these indices
            values = [nums[i] for i in indices]
            values.sort()
            # Assign in ascending order
            for i, val in zip(indices, values):
                ans[i] = val
        
        return ans