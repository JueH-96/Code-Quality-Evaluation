class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        left = 0
        max_len = 0
        count = [0] * 10

        for right in range(n):
            if right > 0 and s[right] == s[right - 1]:
                count[int(s[right])] -= 1
            while sum(count) > 1:
                count[int(s[left])] -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
            count[int(s[right])] += 1

        return max_len