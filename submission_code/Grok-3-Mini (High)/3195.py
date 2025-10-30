class Solution:
    def minimumSteps(self, s: str) -> int:
        ans = 0
        ones = 0
        for char in s:
            if char == '1':
                ones += 1
            else:
                ans += ones
        return ans