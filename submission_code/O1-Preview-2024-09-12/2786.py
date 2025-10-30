class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        left = 0
        max_len = 0
        repeats = 0
        
        for right in range(n):
            if right > 0 and s[right] == s[right - 1]:
                repeats += 1
            while repeats > 1:
                if left + 1 < n and s[left] == s[left + 1]:
                    repeats -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len