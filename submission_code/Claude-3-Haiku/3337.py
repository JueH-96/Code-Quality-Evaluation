class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        for i in range(len(s)):
            left, right = i, i
            while left >= 0 and right < len(s) and s[left] == c and s[right] == c:
                count += 1
                left -= 1
                right += 1
        return count