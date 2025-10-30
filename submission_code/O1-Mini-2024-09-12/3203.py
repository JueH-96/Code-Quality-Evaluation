from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        # Create prefix sum arrays for first half and second half
        prefix_first = [[0] * 26 for _ in range(half + 1)]
        prefix_second = [[0] * 26 for _ in range(half + 1)]
        
        for i in range(half):
            for j in range(26):
                prefix_first[i + 1][j] = prefix_first[i][j]
            prefix_first[i + 1][ord(s[i]) - ord('a')] += 1
            
        for i in range(half, n):
            for j in range(26):
                prefix_second[i - half + 1][j] = prefix_second[i - half][j]
            prefix_second[i - half + 1][ord(s[i]) - ord('a')] += 1
        
        res = []
        for a, b, c, d in queries:
            count_first = [prefix_first[b + 1][j] - prefix_first[a][j] for j in range(26)]
            count_second = [prefix_second[d - half + 1][j] - prefix_second[c - half][j] for j in range(26)]
            total = [count_first[j] + count_second[j] for j in range(26)]
            odd = sum(1 for cnt in total if cnt % 2 != 0)
            res.append(odd <= 1)
        return res