class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        zeros = [0] * (n + 1)
        ones = [0] * (n + 1)
        for i in range(1, n + 1):
            if s[i - 1] == '0':
                zeros[i] = zeros[i - 1] + 1
                ones[i] = ones[i - 1]
            else:
                zeros[i] = zeros[i - 1]
                ones[i] = ones[i - 1] + 1
        count = 0
        for i in range(1, n + 1):
            if ones[i] >= zeros[i] ** 2:
                count += ones[i] - zeros[i] ** 2 + 1
        return count