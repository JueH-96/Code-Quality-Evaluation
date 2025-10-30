class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = 0
        
        for i in range(n):
            zero_count = 0
            one_count = 0
            for j in range(i, n):
                if s[j] == '0':
                    zero_count += 1
                else:
                    one_count += 1
                if zero_count <= k or one_count <= k:
                    total += 1
        return total