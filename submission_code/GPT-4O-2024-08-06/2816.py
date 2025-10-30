class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert the string to a list to allow modifications
        s_list = list(s)
        n = len(s_list)
        
        # Iterate over the first half of the string
        for i in range(n // 2):
            # Compare characters from the start and end
            left_char = s_list[i]
            right_char = s_list[n - i - 1]
            
            # If they are not the same, choose the lexicographically smaller one
            if left_char != right_char:
                # Replace both with the smaller character
                smaller_char = min(left_char, right_char)
                s_list[i] = smaller_char
                s_list[n - i - 1] = smaller_char
        
        # Join the list back into a string
        return ''.join(s_list)