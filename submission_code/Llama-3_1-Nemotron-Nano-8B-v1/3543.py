class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        prefix0 = [0] * (n + 1)
        for i in range(n):
            prefix0[i+1] = prefix0[i] + (1 if s[i] == '0' else 0)
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                zeros = prefix0[j+1] - prefix0[i]
                length = j - i + 1
                ones = length - zeros
                if zeros <= k or ones <= k:
                    count += 1
        return count