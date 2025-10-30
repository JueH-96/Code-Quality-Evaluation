class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        count = 0
        for i in range(len(s)):
            if s[i] == '1':
                count += min(i, len(s) - i - 1) // k + 1
        return count