from collections import Counter

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                counts = Counter(substring)
                is_valid = True
                for count in counts.values():
                    if count > 2:
                        is_valid = False
                        break
                if is_valid:
                    max_length = max(max_length, len(substring))
        return max_length