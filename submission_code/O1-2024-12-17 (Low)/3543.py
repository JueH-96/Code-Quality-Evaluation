class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        
        # Build prefix sums for zeros and ones
        prefix_zeros = [0] * (n + 1)
        prefix_ones = [0] * (n + 1)
        for i in range(n):
            prefix_zeros[i + 1] = prefix_zeros[i] + (1 if s[i] == '0' else 0)
            prefix_ones[i + 1] = prefix_ones[i] + (1 if s[i] == '1' else 0)
        
        count = 0
        
        # Check all substrings
        for start in range(n):
            for end in range(start, n):
                zero_count = prefix_zeros[end + 1] - prefix_zeros[start]
                one_count = prefix_ones[end + 1] - prefix_ones[start]
                
                if zero_count <= k or one_count <= k:
                    count += 1

        return count