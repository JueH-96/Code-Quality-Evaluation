class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        if not s:
            return 0
        left = 0
        bad_pairs = 0
        max_length = 0
        for right in range(len(s)):
            if right > 0 and s[right] == s[right - 1]:
                bad_pairs += 1
            while bad_pairs > 1:
                if left < len(s) - 1 and s[left] == s[left + 1]:
                    bad_pairs -= 1
                left += 1
            max_length = max(max_length, right - left + 1)
        return max_length