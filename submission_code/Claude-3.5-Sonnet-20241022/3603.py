class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        # Create adjacency list representation of the tree
        adj = [[] for _ in range(n)]
        for i in range(1, n):
            adj[parent[i]].append(i)
        # Sort children in increasing order
        for i in range(n):
            adj[i].sort()
        
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]
        
        def dfs(node: int, dfsStr: list) -> None:
            # First visit all children in increasing order
            for child in adj[node]:
                dfs(child, dfsStr)
            # Then add current node's character
            dfsStr.append(s[node])
        
        answer = [False] * n
        
        # For each node as root
        for i in range(n):
            # Use list for string building (more efficient than string concatenation)
            dfsStr = []
            dfs(i, dfsStr)
            # Convert list to string and check if palindrome
            answer[i] = is_palindrome(''.join(dfsStr))
        
        return answer