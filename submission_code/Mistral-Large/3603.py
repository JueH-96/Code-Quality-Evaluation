from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        tree = [[] for _ in range(n)]

        for i in range(1, n):
            tree[parent[i]].append(i)

        def dfs(x):
            global dfsStr
            for y in sorted(tree[x]):
                dfs(y)
            dfsStr += s[x]

        answer = [False] * n
        for i in range(n):
            global dfsStr
            dfsStr = ""
            dfs(i)
            answer[i] = (dfsStr == dfsStr[::-1])

        return answer