class Solution:
    def minimumSteps(self, s: str) -> int:
        zeros = []
        for idx, c in enumerate(s):
            if c == '0':
                zeros.append(idx)
        total = 0
        for i in range(len(zeros)):
            total += zeros[i] - i
        return total