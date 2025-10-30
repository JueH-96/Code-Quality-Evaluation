class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        from collections import defaultdict
        
        # Build the tree from the parent array
        tree = defaultdict(list)
        for i in range(1, len(parent)):
            tree[parent[i]].append(i)
        
        # Function to perform DFS and collect the string in postorder
        def dfs(node):
            nonlocal dfsStr
            # First visit all children in increasing order
            for child in sorted(tree[node]):
                dfs(child)
            # Then add the current node's character
            dfsStr += s[node]
        
        # Check if a string is a palindrome
        def is_palindrome(string):
            return string == string[::-1]
        
        n = len(parent)
        answer = [False] * n
        
        # For each node, perform DFS and check if the resulting string is a palindrome
        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = is_palindrome(dfsStr)
        
        return answer