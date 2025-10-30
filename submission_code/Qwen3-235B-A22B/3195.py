class Solution:
    def minimumSteps(self, s: str) -> int:
        count_ones = 0
        ans = 0
        for c in s:
            if c == '1':
                count_ones += 1
            else:
                ans += count_ones
        return ans