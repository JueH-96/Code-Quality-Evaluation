class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        from collections import defaultdict
        
        # Build the tree from the parent array
        tree = defaultdict(list)
        for child, par in enumerate(parent):
            if par != -1:
                tree[par].append(child)
        
        # Function to perform DFS and build the dfsStr
        def dfs(x):
            nonlocal dfsStr
            for y in sorted(tree[x]):
                dfs(y)
            dfsStr += s[x]
        
        # Function to check if a string is a palindrome
        def is_palindrome(string):
            return string == string[::-1]
        
        n = len(parent)
        answer = [False] * n
        
        # Iterate over each node and check if the dfsStr is a palindrome
        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = is_palindrome(dfsStr)
        
        return answer