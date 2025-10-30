class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = 0
        for i in range(n):
            for j in range(i, n):
                substr = s[i:j+1]
                zero = substr.count('0')
                one = substr.count('1')
                if zero <= k or one <= k:
                    total += 1
        return total