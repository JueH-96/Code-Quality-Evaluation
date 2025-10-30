class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # We'll generate all contiguous substrings (including the empty string)
        subs_s = [""]
        for i in range(len(s)):
            for j in range(i, len(s)):
                subs_s.append(s[i:j+1])
        
        subs_t = [""]
        for i in range(len(t)):
            for j in range(i, len(t)):
                subs_t.append(t[i:j+1])
                
        # Now check every pair (substring from s and substring from t)
        best = 0
        for sub_s in subs_s:
            for sub_t in subs_t:
                candidate = sub_s + sub_t
                # Check if candidate is a palindrome (empty string is a palindrome too)
                if candidate == candidate[::-1]:
                    best = max(best, len(candidate))
        return best

# Some quick tests:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    print(sol.longestPalindrome("a", "a"))       # Expected output: 2
    # Example 2
    print(sol.longestPalindrome("abc", "def"))   # Expected output: 1
    # Example 3
    print(sol.longestPalindrome("b", "aaaa"))    # Expected output: 4
    # Example 4
    print(sol.longestPalindrome("abcde", "ecdba")) # Expected output: 5