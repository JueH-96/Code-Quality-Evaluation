class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)

        ans = [False] * n

        def dfs(u, dfs_str):
            for v in children[u]:
                dfs(v, dfs_str)
            dfs_str.append(s[u])

        for i in range(n):
            dfs_str = []
            dfs(i, dfs_str)
            ans[i] = dfs_str == dfs_str[::-1]

        return ans