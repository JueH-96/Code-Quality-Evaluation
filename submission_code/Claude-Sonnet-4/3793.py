class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def is_palindrome(string):
            return string == string[::-1]
        
        max_length = 0
        
        # Try all possible substrings from s
        for i in range(len(s) + 1):  # +1 to include empty substring
            for j in range(i, len(s) + 1):
                s_sub = s[i:j]
                
                # Try all possible substrings from t
                for k in range(len(t) + 1):  # +1 to include empty substring
                    for l in range(k, len(t) + 1):
                        t_sub = t[k:l]
                        
                        # Concatenate and check if palindrome
                        combined = s_sub + t_sub
                        if is_palindrome(combined):
                            max_length = max(max_length, len(combined))
        
        return max_length