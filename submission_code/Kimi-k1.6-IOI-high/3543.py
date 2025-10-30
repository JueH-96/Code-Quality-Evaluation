class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        for i in range(n):
            count0 = 0
            count1 = 0
            for j in range(i, n):
                if s[j] == '0':
                    count0 += 1
                else:
                    count1 += 1
                if count0 <= k or count1 <= k:
                    result += 1
        return result