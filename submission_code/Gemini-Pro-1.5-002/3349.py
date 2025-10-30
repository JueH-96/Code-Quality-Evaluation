class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_length = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                substring = s[i:j+1]
                valid = True
                for char in set(substring):
                    if substring.count(char) > 2:
                        valid = False
                        break
                if valid:
                    max_length = max(max_length, len(substring))
        return max_length