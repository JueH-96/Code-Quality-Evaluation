class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        max_len = 0

        # Helper function to check if a string is a palindrome
        def is_palindrome(text: str) -> bool:
            return text == text[::-1]

        # List to store all non-empty substrings of s
        s_substrings = []
        for i in range(len(s)):
            for j in range(i, len(s)): # j is the end index (inclusive)
                sub = s[i : j+1]
                s_substrings.append(sub)
                
                # Case 1: Palindrome entirely from s
                if is_palindrome(sub):
                    max_len = max(max_len, len(sub))

        # List to store all non-empty substrings of t
        t_substrings = []
        for i in range(len(t)):
            for j in range(i, len(t)): # j is the end index (inclusive)
                sub = t[i : j+1]
                t_substrings.append(sub)
                
                # Case 2: Palindrome entirely from t
                if is_palindrome(sub):
                    max_len = max(max_len, len(sub))
        
        # Constraints: 1 <= s.length, t.length.
        # This guarantees that s_substrings and t_substrings will contain at least
        # single-character strings (e.g., s[0], t[0]).
        # Since single characters are palindromes, max_len will be at least 1 
        # after the above loops.

        # Case 3: Palindrome formed by concatenating a non-empty substring from s
        # and a non-empty substring from t.
        for sub_s_val in s_substrings:
            for sub_t_val in t_substrings:
                concatenated_string = sub_s_val + sub_t_val
                if is_palindrome(concatenated_string):
                    max_len = max(max_len, len(concatenated_string))
        
        return max_len