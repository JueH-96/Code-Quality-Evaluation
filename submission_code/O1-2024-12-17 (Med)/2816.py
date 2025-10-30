class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        left, right = 0, n - 1
        
        while left < right:
            if s_list[left] != s_list[right]:
                # Change both to the lexicographically smaller character
                smaller_char = min(s_list[left], s_list[right])
                s_list[left] = smaller_char
                s_list[right] = smaller_char
            left += 1
            right -= 1
        
        return "".join(s_list)