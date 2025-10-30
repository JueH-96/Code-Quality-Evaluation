from collections import Counter

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            char_counts = {}
            for j in range(i, n):
                char = s[j]
                char_counts[char] = char_counts.get(char, 0) + 1

                is_valid = False
                for freq in char_counts.values():
                    if freq >= k:
                        is_valid = True
                        break
                if is_valid:
                    count += 1
        return count