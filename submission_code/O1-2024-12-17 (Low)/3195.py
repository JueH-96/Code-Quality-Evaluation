class Solution:
    def minimumSteps(self, s: str) -> int:
        count_ones = 0
        steps = 0
        for char in s:
            if char == '1':
                count_ones += 1
            else:  # char == '0'
                steps += count_ones
        return steps