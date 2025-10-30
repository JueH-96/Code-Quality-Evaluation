class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)

        def is_palindrome(string):
            return string == string[::-1]

        def dfs(x, dfs_str):
            for child in sorted(children[x]):
                dfs_str = dfs(child, dfs_str)
            dfs_str += s[x]
            return dfs_str

        answer = []
        for i in range(n):
            dfs_str = ""
            result_str = dfs(i, dfs_str)
            answer.append(is_palindrome(result_str))

        return answer