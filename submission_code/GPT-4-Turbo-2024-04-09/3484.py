class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert the string to a list of characters for easier manipulation
        chars = list(s)
        n = len(chars)
        
        # Function to check if two digits have the same parity
        def same_parity(a, b):
            return (int(a) % 2) == (int(b) % 2)
        
        # Try to find the first possible swap that results in a lexicographically smaller string
        for i in range(n - 1):
            # Check if the current and next characters have the same parity
            if same_parity(chars[i], chars[i + 1]):
                # Check if swapping them will result in a smaller string
                if chars[i] > chars[i + 1]:
                    # Perform the swap
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    # Return the new string after the swap
                    return ''.join(chars)
        
        # If no swap was made, return the original string
        return s