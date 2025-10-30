class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                zeros = sub.count('0')
                ones = sub.count('1')
                if zeros <= k or ones <= k:
                    count += 1
        return count