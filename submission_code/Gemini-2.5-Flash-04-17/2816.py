class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert the string to a list of characters to make it mutable.
        s_list = list(s)
        n = len(s)
        
        # Use two pointers, one starting from the beginning and one from the end.
        left, right = 0, n - 1
        
        # Iterate until the pointers meet or cross.
        # This covers all symmetric pairs in the string.
        while left < right:
            # Get characters at the current left and right pointers.
            left_char = s_list[left]
            right_char = s_list[right]
            
            # If the characters at the symmetric positions are different,
            # they need to be made equal to satisfy the palindrome property.
            # To achieve the lexicographically smallest palindrome with the minimum number of operations,
            # we change both characters to the minimum of the two differing characters.
            # This counts as one effective operation for this pair and minimizes the character value.
            if left_char != right_char:
                min_char = min(left_char, right_char)
                s_list[left] = min_char
                s_list[right] = min_char
                
            # Move the pointers towards the center.
            # The left pointer moves right, and the right pointer moves left.
            left += 1
            right -= 1
            
        # After the loop, the list of characters represents the resulting palindrome.
        # Join the list back into a string to return.
        return "".join(s_list)