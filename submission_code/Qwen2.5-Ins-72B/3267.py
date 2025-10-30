class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        # Dictionary to store the lengths of special substrings
        special_substrings = defaultdict(int)
        
        n = len(s)
        max_length = -1
        
        # Iterate over all possible substrings
        for i in range(n):
            for j in range(i, n):
                # Check if the substring is special
                if s[i] * (j - i + 1) == s[i:j+1]:
                    special_substrings[s[i:j+1]] += 1
                    # Check if the special substring occurs at least thrice
                    if special_substrings[s[i:j+1]] >= 3:
                        max_length = max(max_length, j - i + 1)
        
        return max_length