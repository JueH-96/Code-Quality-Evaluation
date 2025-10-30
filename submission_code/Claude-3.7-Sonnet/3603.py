class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        # Create adjacency list to represent the tree
        tree = [[] for _ in range(n)]
        for i in range(1, n):  # Start from 1 since parent[0] = -1
            tree[parent[i]].append(i)
        
        answer = [False] * n
        
        def dfs(node, dfsStr):
            # Process children first in increasing order
            for child in tree[node]:
                dfs(child, dfsStr)
            # Add the character at the current node
            dfsStr.append(s[node])
        
        for i in range(n):
            dfsStr = []  # Empty string for each starting node
            dfs(i, dfsStr)
            # Check if the resulting string is a palindrome
            palindrome_str = ''.join(dfsStr)
            answer[i] = palindrome_str == palindrome_str[::-1]
        
        return answer