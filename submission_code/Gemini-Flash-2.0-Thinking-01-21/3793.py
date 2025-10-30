class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)
        max_len = 0

        # Helper function to check if a string is a palindrome
        def is_palindrome(st):
            return st == st[::-1]

        # Iterate through all possible substrings of s: s[i:j] for 0 <= i <= j <= n
        # This covers empty string (i=j) and all non-empty substrings.
        for i in range(n + 1):
            for j in range(i, n + 1):
                sub_s = s[i:j]

                # Iterate through all possible substrings of t: t[k:l] for 0 <= k <= l <= m
                # This covers empty string (k=l) and all non-empty substrings.
                for k in range(m + 1):
                    for l in range(k, m + 1):
                        sub_t = t[k:l]

                        # Concatenate the substrings
                        combined = sub_s + sub_t

                        # Check if the combined string is a palindrome
                        if is_palindrome(combined):
                            # Update the maximum length found so far
                            max_len = max(max_len, len(combined))

        # Since s and t have length at least 1, any single character substring from s or t is a palindrome of length 1.
        # These cases are covered by the loops (e.g., s[i:i+1] with empty t substring, or empty s substring with t[k:k+1]).
        # Thus, max_len will be at least 1 because s and t are non-empty.
        # If the only possible palindrome is the empty string (len 0), max_len would correctly be 0.
        # But constraints guarantee len >= 1, so at least single chars are possible.
        
        return max_len