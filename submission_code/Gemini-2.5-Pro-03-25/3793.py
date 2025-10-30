import math

class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        """
        Finds the length of the longest palindrome formed by concatenating
        a substring of s (possibly empty) and a substring of t (possibly empty).

        Args:
            s: The first input string.
            t: The second input string.

        Returns:
            The length of the longest palindrome that can be formed.

        Constraints:
            1 <= s.length, t.length <= 30
            s and t consist of lowercase English letters.
        """
        n = len(s)
        m = len(t)
        max_len = 0

        # Helper function to check if a string is a palindrome.
        # This function handles empty strings ("" is a palindrome) and single-character strings.
        def is_palindrome(text: str) -> bool:
            # Using slicing for conciseness and clarity in Python.
            # It's generally efficient for strings of this constrained size.
            return text == text[::-1]
            
            # Alternative using two pointers (potentially slightly more efficient by avoiding temporary string creation):
            # len_text = len(text)
            # if len_text <= 1:
            #     return True
            # left, right = 0, len_text - 1
            # while left < right:
            #     if text[left] != text[right]:
            #         return False
            #     left += 1
            #     right -= 1
            # return True


        # Iterate through all possible substrings of s.
        # A substring s[i:j] includes characters from index i up to (but not including) index j.
        # The loops range from 0 to n (inclusive) to handle substrings starting/ending at boundaries
        # and the empty substring case (when i == j).
        for i in range(n + 1):
            for j in range(i, n + 1):
                sub_s = s[i:j]
                
                # Iterate through all possible substrings of t.
                # Similar logic for substrings of t, t[k:l].
                for k in range(m + 1):
                    for l in range(k, m + 1):
                        sub_t = t[k:l]
                        
                        # Concatenate the selected substring from s and the substring from t.
                        p = sub_s + sub_t
                        
                        # Optimization: Only check if the concatenated string is a palindrome
                        # if its length is greater than the maximum length found so far.
                        # This avoids redundant checks for shorter strings.
                        current_len = len(p)
                        if current_len > max_len:
                            # Check if the resulting string 'p' is a palindrome.
                            if is_palindrome(p):
                                # If it is a palindrome, update the maximum length found.
                                max_len = current_len
        
        # After checking all combinations, return the maximum length found.
        return max_len