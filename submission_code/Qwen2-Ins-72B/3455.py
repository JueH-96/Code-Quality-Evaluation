class Solution:
    def minimumLength(self, s: str) -> int:
        left, right = 0, len(s) - 1
        while left < right and s[left] == s[right]:
            curr_char = s[left]
            while left <= right and s[left] == curr_char:
                left += 1
            while left <= right and s[right] == curr_char:
                right -= 1
        return max(0, right - left + 1)