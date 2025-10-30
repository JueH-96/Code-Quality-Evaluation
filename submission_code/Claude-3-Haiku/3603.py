class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        answer = [False] * n
        dfsStr = ""

        def dfs(x):
            nonlocal dfsStr
            for y in range(n):
                if parent[y] == x:
                    dfs(y)
            dfsStr += s[x]

        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = dfsStr == dfsStr[::-1]

        return answer