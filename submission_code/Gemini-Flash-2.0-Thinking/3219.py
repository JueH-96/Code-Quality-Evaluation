class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        dsu = DSU(n)

        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= limit:
                    dsu.union(i, j)

        groups = {}
        for i in range(n):
            root = dsu.find(i)
            groups.setdefault(root, []).append(i)

        res = [0] * n
        for root, indices in groups.items():
            if indices:
                group_values = [nums[i] for i in indices]
                group_values.sort()
                indices.sort()
                for k in range(len(indices)):
                    res[indices[k]] = group_values[k]

        return res