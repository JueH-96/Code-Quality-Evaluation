class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        left = 0
        pair_count = 0
        for right in range(len(s)):
            if right > 0 and s[right] == s[right - 1]:
                pair_count += 1
            while pair_count > 1 and left < right:
                if s[left] == s[left + 1]:
                    pair_count -= 1
                left += 1
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        return max_length