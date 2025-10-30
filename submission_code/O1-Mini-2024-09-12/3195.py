class Solution:
    def minimumSteps(self, s: str) -> int:
        ones = 0
        swaps = 0
        for c in s:
            if c == '1':
                ones += 1
            elif c == '0':
                swaps += ones
        return swaps