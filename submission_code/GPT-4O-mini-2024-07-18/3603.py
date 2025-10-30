class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        from collections import defaultdict
        
        n = len(parent)
        tree = defaultdict(list)
        
        # Build the tree from the parent array
        for child in range(1, n):
            tree[parent[child]].append(child)
        
        # Function to check if a string is a palindrome
        def is_palindrome(string):
            return string == string[::-1]
        
        # Result array
        answer = [False] * n
        
        # DFS function
        def dfs(x):
            nonlocal dfsStr
            # Iterate over children in increasing order
            for child in sorted(tree[x]):
                dfs(child)
            # Add the character of the current node
            dfsStr += s[x]
        
        # Check for each node
        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = is_palindrome(dfsStr)
        
        return answer