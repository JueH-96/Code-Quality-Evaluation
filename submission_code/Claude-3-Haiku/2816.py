class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        left, right = 0, n - 1
        result = list(s)
        
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            elif s[left] < s[right]:
                result[right] = s[left]
                left += 1
                right -= 1
            else:
                result[left] = s[right]
                left += 1
                right -= 1
        
        return ''.join(result)