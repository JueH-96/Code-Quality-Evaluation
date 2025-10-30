class Solution:
    def minimumSteps(self, s: str) -> int:
        zero_count = s.count('0')
        steps = 0
        for char in s:
            if char == '1':
                steps += zero_count
            else:
                zero_count -= 1
        return steps