class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def is_palindrome(text):
            return text == text[::-1]
        
        n, m = len(s), len(t)
        max_length = 0
        
        # Try all possible substrings from s and t
        for i in range(n + 1):
            for j in range(i + 1):
                s_substr = s[j:i]  # Get substring from s
                for k in range(m + 1):
                    for l in range(k + 1):
                        t_substr = t[l:k]  # Get substring from t
                        concat = s_substr + t_substr
                        if is_palindrome(concat):
                            max_length = max(max_length, len(concat))
        
        return max_length