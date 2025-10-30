class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Convert the string to a list of characters
        s = list(s)
        
        # Reverse the list
        s.reverse()
        
        # Initialize a flag to check if we have encountered an odd number
        flag = False
        
        # Iterate over the list
        for i in range(len(s)):
            # If the current character is '1'
            if s[i] == '1':
                # If the previous character is also '1'
                if i > 0 and s[i-1] == '1':
                    # Swap the current character and the previous character
                    s[i], s[i-1] = s[i-1], s[i]
                    # Set the flag to True
                    flag = True
                    # Break the loop
                    break
        
        # If we have encountered an odd number
        if flag:
            # Reverse the list back to its original order
            s.reverse()
            # Join the characters in the list to form a string and return it
            return ''.join(s)
        else:
            # If we have not encountered an odd number, return the original string
            return '1' + ''.join(s[1:])