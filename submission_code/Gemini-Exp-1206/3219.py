class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        num_index = sorted([(num, i) for i, num in enumerate(nums)])
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

        for i in range(1, n):
            if num_index[i][0] - num_index[i - 1][0] <= limit:
                union(num_index[i][1], num_index[i - 1][1])

        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(nums[i])

        for root in groups:
            groups[root].sort()

        group_index = {}
        result = [0] * n
        for i in range(n):
            root = find(i)
            if root not in group_index:
                group_index[root] = 0
            result[i] = groups[root][group_index[root]]
            group_index[root] += 1

        return result