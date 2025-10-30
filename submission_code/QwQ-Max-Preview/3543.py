class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        total = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substr = s[i:j+1]
                cnt0 = substr.count('0')
                cnt1 = len(substr) - cnt0
                if cnt0 <= k or cnt1 <= k:
                    total += 1
        return total