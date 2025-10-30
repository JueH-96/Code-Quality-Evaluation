from typing import List
from collections import defaultdict

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # Prefix sum arrays for character counts
        prefix_count = [defaultdict(int) for _ in range(half + 1)]
        for i in range(half):
            for c in range(97, 123):  # ASCII values for 'a' to 'z'
                prefix_count[i + 1][chr(c)] = prefix_count[i][chr(c)]
            prefix_count[i + 1][s[i]] += 1
        
        def get_count(start, end):
            count = defaultdict(int)
            for c in range(97, 123):
                count[chr(c)] = prefix_count[end + 1][chr(c)] - prefix_count[start][chr(c)]
            return count
        
        def can_form_palindrome(a, b, c, d):
            c, d = n - d - 1, n - c - 1
            if b < c:
                left_count = get_count(a, b)
                mid_count = get_count(b + 1, c - 1)
                right_count = get_count(c, d)
            else:
                left_count = get_count(a, c - 1)
                mid_count = get_count(c, min(b, d))
                right_count = get_count(max(b + 1, d + 1), d)
            
            # Check if we can form a palindrome
            for c in range(97, 123):
                char = chr(c)
                if left_count[char] + right_count[char] != prefix_count[half][char] * 2:
                    return False
                if mid_count[char] % 2 == 1 and left_count[char] + right_count[char] != prefix_count[half][char] * 2 - 1:
                    return False
            
            return True
        
        result = []
        for a, b, c, d in queries:
            result.append(can_form_palindrome(a, b, c, d))
        
        return result