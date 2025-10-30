class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        arr = sorted([(nums[i], i) for i in range(n)])
        res = [0] * n
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            group = []
            group.append(i)
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and arr[j][0] - arr[group[0]][0] <= limit:
                    group.append(j)
                    visited[j] = True
            group_vals = sorted([arr[k][0] for k in group])
            group_indices = sorted([arr[k][1] for k in group])
            for k in range(len(group)):
                res[group_indices[k]] = group_vals[k]
        return res