class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        adj = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:
                adj[parent[i]].append(i)

        answer = [False] * n

        def is_palindrome(text):
            return text == text[::-1]

        def get_dfs_string(start_node):
            dfsStr = ""

            def dfs(u):
                nonlocal dfsStr
                children = sorted(adj[u])
                for v in children:
                    dfs(v)
                dfsStr += s[u]

            dfs(start_node)
            return dfsStr

        for i in range(n):
            dfs_string = get_dfs_string(i)
            answer[i] = is_palindrome(dfs_string)

        return answer