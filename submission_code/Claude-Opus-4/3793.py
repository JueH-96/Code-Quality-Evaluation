class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def isPalindrome(string):
            return string == string[::-1]
        
        max_length = 0
        
        # Try all possible substrings from s (including empty string)
        for i in range(len(s) + 1):
            for j in range(i, len(s) + 1):
                s_substring = s[i:j]
                
                # Try all possible substrings from t (including empty string)
                for k in range(len(t) + 1):
                    for l in range(k, len(t) + 1):
                        t_substring = t[k:l]
                        
                        # Concatenate and check if palindrome
                        combined = s_substring + t_substring
                        if isPalindrome(combined):
                            max_length = max(max_length, len(combined))
        
        return max_length