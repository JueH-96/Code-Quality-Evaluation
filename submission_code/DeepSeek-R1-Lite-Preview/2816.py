class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert the string to a list for mutable operations
        s_list = list(s)
        n = len(s_list)
        
        # Iterate through the first half of the string
        for i in range(n // 2):
            j = n - 1 - i  # Corresponding position from the end
            if s_list[i] != s_list[j]:
                if s_list[i] < s_list[j]:
                    s_list[j] = s_list[i]
                else:
                    s_list[i] = s_list[j]
        
        # Join the list back into a string and return
        return ''.join(s_list)