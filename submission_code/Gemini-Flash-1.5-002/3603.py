class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i, p in enumerate(parent):
            if p != -1:
                children[p].append(i)

        def dfs(x, dfsStr):
            for child in children[x]:
                dfs(child, dfsStr)
            dfsStr.append(s[x])

        answer = []
        for i in range(n):
            dfsStr = []
            dfs(i, dfsStr)
            dfsStr_str = "".join(dfsStr)
            answer.append(dfsStr_str == dfsStr_str[::-1])
        return answer