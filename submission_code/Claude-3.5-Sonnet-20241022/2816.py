class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        chars = list(s)
        n = len(chars)
        
        for i in range(n//2):
            if chars[i] != chars[n-1-i]:
                # Take lexicographically smaller character
                chars[i] = chars[n-1-i] = min(chars[i], chars[n-1-i])
                
        return ''.join(chars)