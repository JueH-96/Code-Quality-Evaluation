class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        from collections import defaultdict

        # Disjoint Set (Union-Find) implementation
        class DSU:
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
                    if self.rank[rx] > self.rank[ry]:
                        self.parent[ry] = rx
                    elif self.rank[rx] < self.rank[ry]:
                        self.parent[rx] = ry
                    else:
                        self.parent[ry] = rx
                        self.rank[rx] += 1
        
        n = len(nums)
        if n <= 1:
            return nums  # trivially smallest

        # Pair each number with its index, then sort by the number's value
        arr = [(val, idx) for idx, val in enumerate(nums)]
        arr.sort(key=lambda x: x[0])

        # Initialize DSU
        dsu = DSU(n)

        # Link consecutive elements in sorted order if their difference <= limit
        for i in range(n - 1):
            if arr[i+1][0] - arr[i][0] <= limit:
                dsu.union(arr[i+1][1], arr[i][1])

        # Group indices by their DSU representative
        groups = defaultdict(list)
        for i in range(n):
            root = dsu.find(i)
            groups[root].append(i)

        # For each group of indices, sort them, extract their values, sort the values,
        # then place the values back in ascending index order.
        result = [0]*n
        for root, indices in groups.items():
            indices.sort()
            # Gather original values
            values = sorted(nums[i] for i in indices)
            # Assign the sorted values to these indices in ascending order
            for i, v in zip(indices, values):
                result[i] = v

        return result