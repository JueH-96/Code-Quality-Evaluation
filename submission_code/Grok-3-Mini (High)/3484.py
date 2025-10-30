class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert string to list for mutability
        chars = list(s)
        n = len(chars)
        
        # Iterate through each pair of adjacent digits
        for i in range(n - 1):
            digit_left = int(chars[i])
            digit_right = int(chars[i + 1])
            
            # Check if left digit is greater than right and they have the same parity
            if digit_left > digit_right and digit_left % 2 == digit_right % 2:
                # Swap the digits
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                # Return the modified string after swap
                return ''.join(chars)
        
        # If no swap was performed, return the original string
        return s