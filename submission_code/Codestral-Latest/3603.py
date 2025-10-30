class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]

        # Build the tree
        for i in range(1, n):
            children[parent[i]].append(i)

        def dfs(x):
            nonlocal dfsStr
            for y in children[x]:
                dfs(y)
            dfsStr += s[x]

        def is_palindrome(string):
            return string == string[::-1]

        answer = [False] * n
        dfsStr = ""

        for i in range(n):
            dfsStr = ""
            dfs(i)
            answer[i] = is_palindrome(dfsStr)

        return answer