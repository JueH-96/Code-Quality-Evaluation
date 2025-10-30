class Solution:
    def minimumSteps(self, s: str) -> int:
        count_ones = 0
        total_swaps = 0
        for c in s:
            if c == '1':
                count_ones += 1
            else:
                total_swaps += count_ones
        return total_swaps