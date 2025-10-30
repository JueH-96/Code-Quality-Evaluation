class Solution:
    def minimumSteps(self, s: str) -> int:
        total_zero = s.count('0')
        current_zero = 0
        res = 0
        for c in s:
            if c == '1':
                res += (total_zero - current_zero)
            else:
                current_zero += 1
        return res