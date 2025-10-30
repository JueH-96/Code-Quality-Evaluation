from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        answer = []
        
        # Precompute prefix sums for character counts
        prefix_sum = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            for j in range(26):
                prefix_sum[i + 1][j] = prefix_sum[i][j] + (s[i] == chr(ord('a') + j))
        
        def can_rearrange(a, b, c, d):
            # Check if the characters in the two ranges can be rearranged to match each other
            for j in range(26):
                if prefix_sum[b + 1][j] - prefix_sum[a][j] != prefix_sum[d + 1][j] - prefix_sum[c][j]:
                    return False
            return True
        
        for a, b, c, d in queries:
            c, d = n - 1 - d, n - 1 - c  # Convert to the corresponding positions in the first half
            if a > c:
                a, b, c, d = c, d, a, b  # Ensure a <= c
            
            # Check if the characters outside the ranges are already a palindrome
            if not all(s[i] == s[n - 1 - i] for i in range(n) if i < a or i > b or i < c or i > d):
                answer.append(False)
                continue
            
            # Check if the characters in the ranges can be rearranged to match each other
            if can_rearrange(a, b, c, d):
                answer.append(True)
            else:
                answer.append(False)
        
        return answer