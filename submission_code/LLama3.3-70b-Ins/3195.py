class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        ones = s.count('1')
        for i in range(len(s) - ones):
            if s[i] == '1':
                steps += 1
        return steps