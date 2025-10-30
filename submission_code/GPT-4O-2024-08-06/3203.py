from typing import List
from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        results = []

        for a, b, c, d in queries:
            # Count characters in the first half substring
            first_half_count = Counter(s[a:b+1])
            # Count characters in the second half substring
            second_half_count = Counter(s[c:d+1])

            # Count characters in the middle part that cannot be rearranged
            middle_count = Counter(s[b+1:c])

            # Combine counts from both halves and the middle
            combined_count = first_half_count + second_half_count + middle_count

            # Check if the combined string can form a palindrome
            odd_count = sum(1 for count in combined_count.values() if count % 2 != 0)

            # A string can be rearranged into a palindrome if at most one character has an odd count
            results.append(odd_count <= 1)

        return results