class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        count = 0
        for i in range(len(s)):
            zeros = ones = 0
            for j in range(i, len(s)):
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                if zeros <= k and ones <= k:
                    count += 1
                else:
                    break
        return count