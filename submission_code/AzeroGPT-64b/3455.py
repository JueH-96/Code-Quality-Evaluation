class Solution:
    def minimumLength(self, s: str) -> int:
        left, right = 0, len(s) - 1
        while left < right and s[left] == s[right]:
            temp = s[left]
            while left <= right and s[left] == temp:
                left += 1
            while left <= right and s[right] == temp:
                right -= 1
        return max(0, right - left + 1)