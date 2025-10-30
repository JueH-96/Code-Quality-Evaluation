class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        total_substrings = n * (n + 1) // 2
        non_dominant_count = 0
        prefix_zeros = [0] * (n + 1)
        
        for i in range(n):
            prefix_zeros[i + 1] = prefix_zeros[i] + (s[i] == '0')
        
        for i in range(n):
            for j in range(i, n):
                zeros = prefix_zeros[j + 1] - prefix_zeros[i]
                ones = (j - i + 1) - zeros
                if ones < zeros * zeros:
                    non_dominant_count += 1
        
        return total_substrings - non_dominant_count