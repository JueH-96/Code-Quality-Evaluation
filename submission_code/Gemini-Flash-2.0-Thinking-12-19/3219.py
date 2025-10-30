class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        parent = list(range(n))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= limit:
                    union(i, j)

        components = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        result_nums = list(nums)
        for root in components:
            indices = components[root]
            values = [nums[i] for i in indices]
            sorted_values = sorted(values)
            sorted_indices = sorted(indices)
            for i in range(len(indices)):
                result_nums[sorted_indices[i]] = sorted_values[i]

        return result_nums