class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        chars = list(s)
        n = len(chars)
        
        # Use two pointers to make the string a palindrome
        # with the minimum number of changes, choosing the lexicographically
        # smaller character on mismatches.
        left = 0
        right = n - 1
        while left < right:
            if chars[left] != chars[right]:
                # Choose the smaller of the two letters
                smaller_char = min(chars[left], chars[right])
                chars[left] = smaller_char
                chars[right] = smaller_char
            left += 1
            right -= 1
        
        return "".join(chars)