class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        answer = [False] * n
        
        def is_palindrome(text):
            return text == text[::-1]
        
        def dfs(node):
            nonlocal dfsStr
            for child in sorted(children[node]):
                dfs(child)
            dfsStr += s[node]
        
        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = is_palindrome(dfsStr)
        
        return answer