class Solution:
    def minimumSteps(self, s: str) -> int:
        total_zeros = s.count('0')
        zeros_so_far = 0
        res = 0
        for c in s:
            if c == '0':
                zeros_so_far += 1
            else:
                res += (total_zeros - zeros_so_far)
        return res