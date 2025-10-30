class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        zero_count = 0
        for i in range(len(s)):
            if s[i] == '0':
                zero_count += 1
            else:
                steps += zero_count
        return steps