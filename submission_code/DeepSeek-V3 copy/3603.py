from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        answer = [False] * n
        
        def dfs(x):
            nonlocal dfsStr
            for y in sorted(children[x]):
                dfs(y)
            dfsStr += s[x]
        
        for i in range(n):
            dfsStr = ""
            dfs(i)
            if dfsStr == dfsStr[::-1]:
                answer[i] = True
        
        return answer