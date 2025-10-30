class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        dominant_ones_count = 0
        for i in range(n):
            num_zeros = 0
            num_ones = 0
            for j in range(i, n):
                if s[j] == '0':
                    num_zeros += 1
                else:
                    num_ones += 1
                if num_ones >= num_zeros * num_zeros:
                    dominant_ones_count += 1
        return dominant_ones_count