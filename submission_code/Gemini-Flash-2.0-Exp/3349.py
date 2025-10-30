class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_len = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                sub = s[i:j+1]
                counts = {}
                valid = True
                for char in sub:
                    counts[char] = counts.get(char, 0) + 1
                    if counts[char] > 2:
                        valid = False
                        break
                if valid:
                    max_len = max(max_len, len(sub))
        return max_len