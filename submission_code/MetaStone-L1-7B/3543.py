class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        total = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                count0 = substring.count('0')
                count1 = substring.count('1')
                if count0 <= k or count1 <= k:
                    total += 1
        return total