class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        count_zero = 0
        for c in reversed(s):
            if c == '0':
                count_zero += 1
            else:
                steps += count_zero
        return steps