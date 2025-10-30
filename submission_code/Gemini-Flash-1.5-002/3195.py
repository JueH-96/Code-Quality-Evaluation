class Solution:
    def minimumSteps(self, s: str) -> int:
        count_ones = s.count('1')
        steps = 0
        for i in range(count_ones):
            if s[i] == '0':
                steps += 1
        return steps