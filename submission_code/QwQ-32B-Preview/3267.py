from typing import List, Dict
from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        max_len = -1
        # Group consecutive identical characters
        groups = []
        i = 0
        while i < len(s):
            char = s[i]
            length = 1
            while i + length < len(s) and s[i + length] == char:
                length += 1
            groups.append((char, length))
            i += length
        # For each unique character, calculate substring counts
        substring_counts = defaultdict(lambda: defaultdict(int))
        for char, length in groups:
            for k in range(1, length + 1):
                substring_counts[char][k] += (length - k + 1)
        # Find the maximum length with total count >= 3
        for char in substring_counts:
            for k in substring_counts[char]:
                if substring_counts[char][k] >= 3:
                    max_len = max(max_len, k)
        return max_len if max_len != -1 else -1