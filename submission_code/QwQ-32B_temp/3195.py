class Solution:
    def minimumSteps(self, s: str) -> int:
        count_ones = 0
        result = 0
        for c in s:
            if c == '1':
                count_ones += 1
            else:
                result += count_ones
        return result