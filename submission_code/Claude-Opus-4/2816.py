class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        
        for i in range(n // 2):
            left = i
            right = n - 1 - i
            
            if s_list[left] != s_list[right]:
                # Choose the lexicographically smaller character
                if s_list[left] < s_list[right]:
                    s_list[right] = s_list[left]
                else:
                    s_list[left] = s_list[right]
        
        return ''.join(s_list)