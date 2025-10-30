class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        from collections import defaultdict
        
        # Build the tree from the parent array
        tree = defaultdict(list)
        for child, par in enumerate(parent):
            if par != -1:
                tree[par].append(child)
        
        # Function to perform DFS and build dfsStr
        def dfs(node):
            nonlocal dfsStr
            for child in sorted(tree[node]):
                dfs(child)
            dfsStr += s[node]
        
        # Function to check if a string is a palindrome
        def is_palindrome(string):
            return string == string[::-1]
        
        n = len(parent)
        answer = [False] * n
        
        # Check each node as the root of the DFS
        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = is_palindrome(dfsStr)
        
        return answer