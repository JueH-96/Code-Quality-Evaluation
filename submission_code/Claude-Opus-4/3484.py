class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert string to list for easier manipulation
        chars = list(s)
        
        # Check each adjacent pair
        for i in range(len(chars) - 1):
            # Get the two adjacent digits
            digit1 = int(chars[i])
            digit2 = int(chars[i + 1])
            
            # Check if they have the same parity
            if digit1 % 2 == digit2 % 2:
                # Check if swapping would make the string smaller
                if chars[i] > chars[i + 1]:
                    # Swap the digits
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    # Return immediately after the first swap
                    return ''.join(chars)
        
        # If no beneficial swap was found, return the original string
        return s