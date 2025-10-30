from typing import List
from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        mid = n // 2
        
        def is_palindrome_possible(a: int, b: int, c: int, d: int) -> bool:
            # Count characters in the substrings
            left_count = Counter(s[a:b+1])
            right_count = Counter(s[c:d+1])
            
            # Count characters in the middle substring
            middle_count = Counter(s[b+1:c])
            
            # Check if the middle substring can be rearranged to make the string a palindrome
            for char, count in middle_count.items():
                if count % 2 != 0:
                    if left_count[char] == 0 and right_count[char] == 0:
                        return False
                    elif left_count[char] > 0:
                        left_count[char] -= 1
                    elif right_count[char] > 0:
                        right_count[char] -= 1
            
            # Check if the left and right substrings can be rearranged to make the string a palindrome
            for char in left_count:
                if left_count[char] > 0 and right_count[char] == 0:
                    return False
            for char in right_count:
                if right_count[char] > 0 and left_count[char] == 0:
                    return False
            
            return True
        
        result = []
        for query in queries:
            a, b, c, d = query
            result.append(is_palindrome_possible(a, b, c, d))
        
        return result