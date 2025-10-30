class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        # Create an array indicating consecutive duplicates (1 if s[i] == s[i+1], else 0)
        pairs = [0] * (n - 1)
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                pairs[i] = 1
        
        # Create prefix sums of the pairs array
        prefix_sums = [0] * (len(pairs) + 1)
        for i in range(1, len(prefix_sums)):
            prefix_sums[i] = prefix_sums[i - 1] + pairs[i - 1]
        
        max_len = 1  # At least one character is always valid
        
        # Check all possible substrings
        for i in range(n):
            for j in range(i, n):
                # Number of consecutive pairs in substring s[i..j] is prefix_sums[j] - prefix_sums[i]
                current_pairs = prefix_sums[j] - prefix_sums[i]
                if current_pairs <= 1:
                    max_len = max(max_len, j - i + 1)
        
        return max_len