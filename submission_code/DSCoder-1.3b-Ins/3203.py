class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        def check(a, b, c, d):
            return sum([1 for i in range(a, b+1) if s[i] != s[d-c+a]]) == 0

        return [check(a, b, c, d) for a, b, c, d in queries]