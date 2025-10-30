class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                zeros = 0
                ones = 0
                for char in sub:
                    if char == '0':
                        zeros += 1
                    else:
                        ones += 1
                if ones >= zeros * zeros:
                    count += 1
        return count