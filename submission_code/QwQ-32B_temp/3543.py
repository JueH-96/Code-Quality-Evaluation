class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        prefix_zeros = [0] * (n + 1)
        prefix_ones = [0] * (n + 1)
        
        for i in range(n):
            prefix_zeros[i+1] = prefix_zeros[i] + (s[i] == '0')
            prefix_ones[i+1] = prefix_ones[i] + (s[i] == '1')
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                zeros = prefix_zeros[j+1] - prefix_zeros[i]
                ones = prefix_ones[j+1] - prefix_ones[i]
                if zeros <= k or ones <= k:
                    count += 1
        return count