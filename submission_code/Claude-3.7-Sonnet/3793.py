class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        max_length = 0
        
        for i in range(len(s) + 1):  # Starting position in s (including empty)
            for j in range(i, len(s) + 1):  # Ending position in s
                s_sub = s[i:j]
                for k in range(len(t) + 1):  # Starting position in t (including empty)
                    for l in range(k, len(t) + 1):  # Ending position in t
                        t_sub = t[k:l]
                        combined = s_sub + t_sub
                        # Check if non-empty and is palindrome
                        if combined and combined == combined[::-1]:
                            max_length = max(max_length, len(combined))
        
        return max_length