class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Given s and t of max length 30, we can enumerate all substrings of s and t.
        # We want to find the maximum length palindrome that can be formed by concatenating a substring of s (possibly empty)
        # and a substring of t (possibly empty).
        # We'll consider all possible substrings (including the empty substring) and check if the concatenated string is a palindrome.
        # The empty string is a palindrome, but length 0 is not likely to be the maximum answer.
        
        # Helper function to check if a given string is a palindrome.
        def is_palindrome(string: str) -> bool:
            return string == string[::-1]
        
        max_len = 0
        
        # We include empty substring for s: iterate over start from 0 to len(s) and generate substring s[i:j] 
        # where if i==j, that's the empty substring. Similarly for t.
        n, m = len(s), len(t)
        
        # Enumerate over all substrings from s (including empty)
        for i in range(n+1):
            for j in range(i, n+1):
                sub_s = s[i:j]
                # Enumerate all substrings from t (including empty)
                for k in range(m+1):
                    for l in range(k, m+1):
                        sub_t = t[k:l]
                        candidate = sub_s + sub_t
                        # Check if candidate is a palindrome.
                        if is_palindrome(candidate):
                            max_len = max(max_len, len(candidate))
                            
        return max_len