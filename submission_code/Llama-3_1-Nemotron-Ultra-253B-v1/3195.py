class Solution:
    def minimumSteps(self, s: str) -> int:
        count_zeros = 0
        total = 0
        for c in reversed(s):
            if c == '1':
                total += count_zeros
            else:
                count_zeros += 1
        return total