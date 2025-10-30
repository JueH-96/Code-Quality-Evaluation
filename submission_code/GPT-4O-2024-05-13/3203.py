from typing import List
from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        result = []
        
        for a, b, c, d in queries:
            left_part = s[a:b+1]
            right_part = s[c:d+1]
            
            left_counter = Counter(left_part)
            right_counter = Counter(right_part)
            
            # Check if the remaining parts of the string can form a palindrome
            middle_part = s[b+1:c]
            middle_counter = Counter(middle_part)
            
            # Combine the counters
            combined_counter = left_counter + middle_counter + right_counter
            
            # Check if the combined string can be rearranged to form a palindrome
            odd_count = sum(1 for count in combined_counter.values() if count % 2 != 0)
            
            # A string can be rearranged to form a palindrome if at most one character has an odd count
            result.append(odd_count <= 1)
        
        return result