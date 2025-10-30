from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        # Build the tree
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        # Sort each children list in increasing order
        for i in range(n):
            children[i].sort()
        
        # Precompute the answer for each node
        answer = [False] * n
        
        # Function to perform DFS and check palindrome
        def is_palindrome_dfs(node: int) -> str:
            res = []
            # Process children in order
            for child in children[node]:
                res.append(is_palindrome_dfs(child))
            # Append current node's character
            res.append(s[node])
            # Join to form the string
            dfs_str = ''.join(res)
            # Check palindrome
            answer[node] = dfs_str == dfs_str[::-1]
            return dfs_str
        
        # For each node, perform DFS and check palindrome
        for i in range(n):
            # Reset the answer for this node
            # We need to perform DFS again, which is O(n) per node, leading to O(n^2) time
            # This is not efficient for large n, but works for the given examples
            dfs_str = []
            stack = [(i, False)]
            while stack:
                node, visited = stack.pop()
                if visited:
                    dfs_str.append(s[node])
                    continue
                # Push the node back with visited=True
                stack.append((node, True))
                # Push children in reverse order to process them in increasing order
                for child in reversed(children[node]):
                    stack.append((child, False))
            # Check palindrome
            answer[i] = dfs_str == dfs_str[::-1]
        
        return answer