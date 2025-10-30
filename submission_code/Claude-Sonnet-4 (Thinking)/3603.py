class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Build the adjacency list for children
        children = [[] for _ in range(n)]
        for i in range(1, n):  # Start from 1 since node 0 is the root
            children[parent[i]].append(i)
        
        # Sort children for each node to ensure increasing order
        for i in range(n):
            children[i].sort()
        
        def dfs(node):
            result = ""
            # Iterate over children in increasing order
            for child in children[node]:
                result += dfs(child)
            # Add the character of the current node
            result += s[node]
            return result
        
        def is_palindrome(string):
            return string == string[::-1]
        
        answer = []
        for i in range(n):
            result_str = dfs(i)
            answer.append(is_palindrome(result_str))
        
        return answer