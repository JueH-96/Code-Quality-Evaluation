from typing import List
from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2

        # Precompute the frequency of characters for the first half and the second half
        first_half_freq = [Counter() for _ in range(half + 1)]
        second_half_freq = [Counter() for _ in range(half + 1)]

        for i in range(half):
            first_half_freq[i + 1] = first_half_freq[i].copy()
            first_half_freq[i + 1][s[i]] += 1

        for i in range(half, n):
            second_half_freq[i - half + 1] = second_half_freq[i - half].copy()
            second_half_freq[i - half + 1][s[i]] += 1

        results = []

        for a, b, c, d in queries:
            # Frequency counters for the substrings
            first_part = first_half_freq[b + 1] - first_half_freq[a]
            second_part = second_half_freq[d - half + 1] - second_half_freq[c - half]

            # Combine the two parts
            combined = first_part + second_part

            # Count the number of characters with odd frequencies
            odd_count = sum(1 for count in combined.values() if count % 2 != 0)

            # A string can form a palindrome if it has at most one character with an odd frequency
            results.append(odd_count <= 1)

        return results