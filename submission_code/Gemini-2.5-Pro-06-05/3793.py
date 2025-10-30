class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)
        max_len = 0

        # Iterate through all substrings of s, including the empty string.
        # A substring s[i:j] is from index i to j-1.
        for i in range(n + 1):
            for j in range(i, n + 1):
                sub_s = s[i:j]
                
                # Iterate through all substrings of t, including the empty string.
                for k in range(m + 1):
                    for l in range(k, m + 1):
                        sub_t = t[k:l]
                        
                        candidate = sub_s + sub_t
                        
                        # An empty string is a palindrome, but its length is 0.
                        # The problem constraints (lengths >= 1) ensure at least one
                        # character exists, so a palindrome of length at least 1 is
                        # always possible.
                        
                        # Check if the candidate string is a palindrome.
                        # The most Pythonic way is to compare it with its reverse.
                        if candidate and candidate == candidate[::-1]:
                            max_len = max(max_len, len(candidate))
                            
        return max_len