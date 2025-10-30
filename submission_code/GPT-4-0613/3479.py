class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        zeros = [0] * (n + 1)
        for i in range(n):
            zeros[i + 1] = zeros[i] + (s[i] == '0')
        l = 0
        for r in range(n):
            while zeros[r + 1] - zeros[l] > (r - l + 1) ** 0.5:
                l += 1
            total -= r - l + 1
        return total