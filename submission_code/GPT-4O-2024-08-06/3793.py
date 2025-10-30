class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def is_palindrome(s):
            return s == s[::-1]
        
        max_length = 0
        # Check all possible substrings of s
        for i in range(len(s) + 1):
            for j in range(i, len(s) + 1):
                s_sub = s[i:j]
                # Check all possible substrings of t
                for k in range(len(t) + 1):
                    for l in range(k, len(t) + 1):
                        t_sub = t[k:l]
                        combined = s_sub + t_sub
                        if is_palindrome(combined):
                            max_length = max(max_length, len(combined))
        
        return max_length