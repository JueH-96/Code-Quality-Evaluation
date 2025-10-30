class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                char_counts = {}
                valid_substring = True
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1
                for char in char_counts:
                    if char_counts[char] > 2:
                        valid_substring = False
                        break
                if valid_substring:
                    max_length = max(max_length, len(substring))
        return max_length