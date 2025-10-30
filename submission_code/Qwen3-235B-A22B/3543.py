class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        prefix_0 = [0] * (n + 1)
        for i in range(n):
            prefix_0[i + 1] = prefix_0[i] + (s[i] == '0')
        count = 0
        for i in range(n):
            for j in range(i, n):
                zeros = prefix_0[j + 1] - prefix_0[i]
                len_sub = j - i + 1
                ones = len_sub - zeros
                if zeros <= k or ones <= k:
                    count += 1
        return count