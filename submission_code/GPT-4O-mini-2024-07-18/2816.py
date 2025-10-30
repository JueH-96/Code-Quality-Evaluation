class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        s_list = list(s)
        
        for i in range(n // 2):
            left_char = s_list[i]
            right_char = s_list[n - 1 - i]
            # Choose the lexicographically smaller character to form a palindrome
            smallest_char = min(left_char, right_char)
            s_list[i] = smallest_char
            s_list[n - 1 - i] = smallest_char
        
        return ''.join(s_list)