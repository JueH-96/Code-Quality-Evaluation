class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def is_palindrome(string):
            return string == string[::-1]
        
        max_length = 0
        
        # Try all possible combinations of substrings from s and t
        for i in range(len(s) + 1):
            for j in range(i, len(s) + 1):
                for k in range(len(t) + 1):
                    for l in range(k, len(t) + 1):
                        s_part = s[i:j]
                        t_part = t[k:l]
                        combined = s_part + t_part
                        if is_palindrome(combined):
                            max_length = max(max_length, len(combined))
        
        return max_length