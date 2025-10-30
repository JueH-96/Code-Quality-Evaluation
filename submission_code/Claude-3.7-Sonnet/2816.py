class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert to list for easier modification
        s_list = list(s)
        n = len(s)
        
        for i in range(n // 2):
            if s_list[i] != s_list[n-1-i]:
                # Choose the lexicographically smaller character
                if s_list[i] < s_list[n-1-i]:
                    s_list[n-1-i] = s_list[i]
                else:
                    s_list[i] = s_list[n-1-i]
        
        return ''.join(s_list)