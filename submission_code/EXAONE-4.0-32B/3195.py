class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        k = s.count('1')
        j = 0
        total_steps = 0
        for i, char in enumerate(s):
            if char == '1':
                total_steps += (n - k + j - i)
                j += 1
        return total_steps