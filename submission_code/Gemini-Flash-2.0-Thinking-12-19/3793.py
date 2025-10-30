class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        
        def is_palindrome(string):
            # Helper function to check if a string is a palindrome
            return string == string[::-1]

        max_len = 0

        # Iterate over all possible substrings of s
        # A substring s[i:j] goes from index i up to (but not including) index j.
        # To include the empty string and all substrings, i and j should range from 0 to len(s).
        # When i == j, the substring s[i:j] is empty.
        for i in range(len(s) + 1):
            for j in range(i, len(s) + 1):
                sub_s = s[i:j]

                # Iterate over all possible substrings of t
                # Similarly, k and l should range from 0 to len(t).
                # When k == l, the substring t[k:l] is empty.
                for k in range(len(t) + 1):
                    for l in range(k, len(t) + 1):
                        sub_t = t[k:l]

                        # Concatenate the chosen substrings
                        combined = sub_s + sub_t

                        # Check if the concatenated string is a palindrome
                        if is_palindrome(combined):
                            # Update the maximum length found so far
                            max_len = max(max_len, len(combined))

        return max_len