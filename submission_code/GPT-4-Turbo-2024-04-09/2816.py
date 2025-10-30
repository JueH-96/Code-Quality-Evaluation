class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        s = list(s)
        left = 0
        right = n - 1
        
        while left < right:
            if s[left] != s[right]:
                # To make lexicographically smallest, choose the smaller character
                if s[left] < s[right]:
                    s[right] = s[left]
                else:
                    s[left] = s[right]
            left += 1
            right -= 1
        
        return ''.join(s)