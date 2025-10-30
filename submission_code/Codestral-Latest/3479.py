class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0

        for i in range(n):
            ones = 0
            zeros = 0
            for j in range(i, n):
                if s[j] == '1':
                    ones += 1
                else:
                    zeros += 1
                if ones >= zeros * zeros:
                    count += 1

        return count