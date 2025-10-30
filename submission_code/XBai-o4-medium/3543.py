class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        prefix_0 = [0] * (n + 1)
        prefix_1 = [0] * (n + 1)
        
        for i in range(n):
            prefix_0[i+1] = prefix_0[i] + (1 if s[i] == '0' else 0)
            prefix_1[i+1] = prefix_1[i] + (1 if s[i] == '1' else 0)
        
        res = 0
        for i in range(n):
            for j in range(i, n):
                zeros = prefix_0[j+1] - prefix_0[i]
                ones = prefix_1[j+1] - prefix_1[i]
                if zeros <= k or ones <= k:
                    res += 1
        return res