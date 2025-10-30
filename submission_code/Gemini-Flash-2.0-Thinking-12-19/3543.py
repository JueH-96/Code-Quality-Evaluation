class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                zeros = 0
                ones = 0
                for char in substring:
                    if char == '0':
                        zeros += 1
                    else:
                        ones += 1
                if zeros <= k or ones <= k:
                    count += 1
        return count