class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        results = []
        for query in queries:
            a, b, c, d = query[0], query[1], query[2], query[3]
            possible = True
            for i in range(n // 2):
                j = n - 1 - i
                if s[i] != s[j]:
                    if not ((a <= i <= b) or (c <= j <= d)):
                        possible = False
                        break
            results.append(possible)
        return results