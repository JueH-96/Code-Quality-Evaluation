class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        left = 0
        for right in range(len(s)):
            if s[right] == '1':
                steps += right - left
                left += 1
        return steps