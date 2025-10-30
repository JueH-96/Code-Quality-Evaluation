class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        tree = [[] for _ in range(n)]
        for i in range(1, n):
            tree[parent[i]].append(i)
        
        answer = [False] * n
        
        def is_palindrome(string):
            return string == string[::-1]
        
        def dfs(node):
            dfsStr = ""
            for child in sorted(tree[node]):
                dfsStr += dfs(child)
            dfsStr += s[node]
            return dfsStr
        
        for i in range(n):
            result = dfs(i)
            answer[i] = is_palindrome(result)
        
        return answer