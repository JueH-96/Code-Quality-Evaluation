class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        parent = list(range(n))
        size = [1] * n
        limit_map = [[] for _ in range(limit + 1)]
        nums_map = [[] for _ in range(2 * 10**9 + 1)]

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            x, y = find(x), find(y)
            if x == y:
                return False
            if size[x] < size[y]:
                x, y = y, x
            parent[y] = x
            size[x] += size[y]
            return True

        for i, num in enumerate(nums):
            for j in range(max(0, num - limit), num):
                for k in nums_map[j]:
                    union(i, k)
            nums_map[num].append(i)
            for j in limit_map[num]:
                union(i, j)
            limit_map[num].append(i)

        groups = [[] for _ in range(n)]
        for i, num in enumerate(nums):
            groups[find(i)].append(num)

        for group in groups:
            group.sort(reverse=True)

        res = []
        for i in range(n):
            res.append(groups[find(i)].pop())

        return res