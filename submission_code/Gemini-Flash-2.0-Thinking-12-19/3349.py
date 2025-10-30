class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        n = len(s)
        max_len = 0
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                char_counts = {}
                valid_substring = True
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1
                for count in char_counts.values():
                    if count > 2:
                        valid_substring = False
                        break
                if valid_substring:
                    max_len = max(max_len, len(substring))
        return max_len