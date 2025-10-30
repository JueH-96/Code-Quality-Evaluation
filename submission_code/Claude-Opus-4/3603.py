class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Build adjacency list for the tree
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        # Sort children lists to ensure we visit in increasing order
        for i in range(n):
            children[i].sort()
        
        def dfs(node, dfsStr):
            # Visit all children first
            for child in children[node]:
                dfs(child, dfsStr)
            # Then append current node's character
            dfsStr.append(s[node])
        
        def is_palindrome(arr):
            left, right = 0, len(arr) - 1
            while left < right:
                if arr[left] != arr[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        answer = []
        for i in range(n):
            dfsStr = []
            dfs(i, dfsStr)
            answer.append(is_palindrome(dfsStr))
        
        return answer