class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Generate all contiguous substrings of s (including the empty substring)
        subs_s = [""]
        for i in range(len(s)):
            for j in range(i, len(s)):
                subs_s.append(s[i:j+1])
                
        # Generate all contiguous substrings of t (including the empty substring)
        subs_t = [""]
        for i in range(len(t)):
            for j in range(i, len(t)):
                subs_t.append(t[i:j+1])
        
        max_len = 0
        
        # Check each combination (substring from s + substring from t)
        for sub_s in subs_s:
            for sub_t in subs_t:
                candidate = sub_s + sub_t
                # Check palindrome property. An empty string is a palindrome as well,
                # but we are looking for the longest so we update only if it is longer.
                if candidate == candidate[::-1]:
                    max_len = max(max_len, len(candidate))
                    
        return max_len

# You can run the following tests to verify the solution:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.longestPalindrome("a", "a"))         # Expected output: 2
    # Example 2:
    print(sol.longestPalindrome("abc", "def"))       # Expected output: 1
    # Example 3:
    print(sol.longestPalindrome("b", "aaaa"))        # Expected output: 4
    # Example 4:
    print(sol.longestPalindrome("abcde", "ecdba"))   # Expected output: 5