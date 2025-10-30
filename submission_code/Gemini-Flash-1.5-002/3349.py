class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        if not s or len(s) < 2:
            return len(s)

        max_len = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                substring = s[i:j+1]
                char_counts = {}
                valid = True
                for char in substring:
                    char_counts[char] = char_counts.get(char, 0) + 1
                    if char_counts[char] > 2:
                        valid = False
                        break
                if valid:
                    max_len = max(max_len, len(substring))

        return max_len