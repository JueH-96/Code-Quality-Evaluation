class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)

        def is_palindrome(string):
            return string == string[::-1]

        def dfs(node, dfsStr):
            for child in sorted(children[node]):
                dfs(child, dfsStr)
            dfsStr.append(s[node])
            return dfsStr

        answer = [False] * n
        for i in range(n):
            dfsStr = []
            result_str = "".join(dfs(i, dfsStr))
            answer[i] = is_palindrome(result_str)
        
        return answer