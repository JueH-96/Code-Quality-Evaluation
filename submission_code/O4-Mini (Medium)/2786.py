class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_len = 1  # At least one character is always valid
        # Check all substrings
        for i in range(n):
            # We can start with current max length and try to extend
            for j in range(i + max_len, n):
                # Count consecutive equal digit pairs in s[i:j+1]
                count_pairs = 0
                # Only need to scan from i to j-1
                for k in range(i, j):
                    if s[k] == s[k + 1]:
                        count_pairs += 1
                        # Early exit if more than one pair
                        if count_pairs > 1:
                            break
                # If valid semi-repetitive, update max_len
                if count_pairs <= 1:
                    curr_len = j - i + 1
                    if curr_len > max_len:
                        max_len = curr_len
                else:
                    # No need to try longer j for this i if we've exceeded pairs
                    break
        return max_len