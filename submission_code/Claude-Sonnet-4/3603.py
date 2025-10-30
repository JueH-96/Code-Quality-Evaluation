class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Build adjacency list for the tree
        children = [[] for _ in range(n)]
        for i in range(1, n):  # Skip root (index 0)
            children[parent[i]].append(i)
        
        # Sort children lists to ensure we visit in increasing order
        for i in range(n):
            children[i].sort()
        
        def dfs(node):
            dfs_str = []
            
            def dfs_helper(x):
                # Visit children in increasing order
                for child in children[x]:
                    dfs_helper(child)
                # Add current node's character
                dfs_str.append(s[x])
            
            dfs_helper(node)
            return ''.join(dfs_str)
        
        def is_palindrome(string):
            return string == string[::-1]
        
        answer = []
        for i in range(n):
            dfs_result = dfs(i)
            answer.append(is_palindrome(dfs_result))
        
        return answer