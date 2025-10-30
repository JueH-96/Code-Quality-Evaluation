class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        # Convert the string to a list of characters for mutability
        s_list = list(s)
        
        left = 0
        right = n - 1
        
        # Iterate with two pointers, moving inwards from both ends
        while left < right:
            # Compare characters at the current left and right positions
            if s_list[left] != s_list[right]:
                # If they are different, we need to change one of them.
                # To get the lexicographically smallest palindrome,
                # we choose the smaller character between the two and
                # apply it to both positions.
                chosen_char = min(s_list[left], s_list[right])
                s_list[left] = chosen_char
                s_list[right] = chosen_char
            
            # Move pointers inwards for the next pair
            left += 1
            right -= 1
            
        # Join the list of characters back into a string and return it
        return "".join(s_list)