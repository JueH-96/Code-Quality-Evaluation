class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        white_count = 0
        for i in range(len(s)):
            if s[i] == '0':
                steps += white_count
            else:
                white_count += 1
        return steps