class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        
        for i in range(n // 2):
            left = i
            right = n - 1 - i
            
            if s_list[left] != s_list[right]:
                # Choose the lexicographically smaller character
                smaller_char = min(s_list[left], s_list[right])
                s_list[left] = smaller_char
                s_list[right] = smaller_char
        
        return ''.join(s_list)