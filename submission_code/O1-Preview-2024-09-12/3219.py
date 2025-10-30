class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        from collections import defaultdict

        n = len(nums)
        parent = [i for i in range(n)]
        rank = [0]*n

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u,v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return
            if rank[pu] < rank[pv]:
                parent[pu] = pv
            else:
                parent[pv] = pu
                if rank[pu] == rank[pv]:
                    rank[pu] +=1

        nums_with_index = [(nums[i], i) for i in range(n)]
        nums_with_index.sort()

        for i in range(n-1):
            num1, idx1 = nums_with_index[i]
            num2, idx2 = nums_with_index[i+1]
            if num2 - num1 <= limit:
                union(idx1, idx2)

        groups = defaultdict(list)
        for idx in range(n):
            root = find(idx)
            groups[root].append(idx)

        res = [0]*n
        for group_indices in groups.values():
            group_nums = [nums[i] for i in group_indices]
            group_indices.sort()
            group_nums.sort()
            for idx, val in zip(group_indices, group_nums):
                res[idx] = val

        return res