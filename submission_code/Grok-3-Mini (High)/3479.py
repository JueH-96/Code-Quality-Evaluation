class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        # Precompute prefix sums for zeros and ones
        prefix_zeros = [0] * (n + 1)
        prefix_ones = [0] * (n + 1)
        for i in range(n):
            if s[i] == '0':
                prefix_zeros[i + 1] = prefix_zeros[i] + 1
                prefix_ones[i + 1] = prefix_ones[i]
            else:
                prefix_zeros[i + 1] = prefix_zeros[i]
                prefix_ones[i + 1] = prefix_ones[i] + 1
        
        # Iterate over all possible substrings using two pointers
        for j in range(n):
            for i in range(j + 1):  # i from 0 to j inclusive
                num_zeros = prefix_zeros[j + 1] - prefix_zeros[i]
                num_ones = prefix_ones[j + 1] - prefix_ones[i]
                if num_ones >= num_zeros * num_zeros:
                    ans += 1
        
        return ans