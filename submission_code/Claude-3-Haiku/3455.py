class Solution:
    def minimumLength(self, s: str) -> int:
        while len(s) > 1 and s[0] == s[-1]:
            left, right = 0, len(s) - 1
            while left < right and s[left] == s[0]:
                left += 1
            while right > left and s[right] == s[-1]:
                right -= 1
            s = s[left:right+1]
        return len(s)