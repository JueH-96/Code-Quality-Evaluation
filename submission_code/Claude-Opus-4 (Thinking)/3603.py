class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Build adjacency list for children
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        answer = []
        
        for start in range(n):
            dfsStr = []
            
            def dfs(x):
                # Visit children in increasing order
                for child in children[x]:
                    dfs(child)
                # Add current node's character
                dfsStr.append(s[x])
            
            dfs(start)
            
            # Check if palindrome
            is_palindrome = dfsStr == dfsStr[::-1]
            answer.append(is_palindrome)
        
        return answer