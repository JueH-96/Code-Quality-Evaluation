class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def is_palindrome(string):
            return string == string[::-1]
        
        max_len = 0
        n, m = len(s), len(t)
        
        # Try all substrings from s and t
        for i in range(n + 1):
            for j in range(i, n + 1):
                for k in range(m + 1):
                    for l in range(k, m + 1):
                        concatenated = s[i:j] + t[k:l]
                        if is_palindrome(concatenated):
                            max_len = max(max_len, len(concatenated))
        
        return max_len