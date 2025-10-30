class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        
        # Iterate over every possible starting index for the substring.
        for i in range(n):
            consecutive_pair_count = 0  # Count of consecutive identical digit pairs in current substring.
            # Extend the substring one character at a time.
            for j in range(i, n):
                # If this character and the previous one are the same, it's a consecutive pair.
                if j > i and s[j] == s[j - 1]:
                    consecutive_pair_count += 1
                
                # If we have more than one consecutive pair, we break out since further 
                # extension will not make the substring valid.
                if consecutive_pair_count > 1:
                    break
                
                # Otherwise, update the maximum length.
                max_length = max(max_length, j - i + 1)
        
        return max_length