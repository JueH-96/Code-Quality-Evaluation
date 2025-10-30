class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert string to list for easy manipulation
        chars = list(s)
        n = len(chars)
        
        for i in range(n - 1):
            # Check if current and next digits have the same parity
            current = int(chars[i])
            next_char = int(chars[i+1])
            if (current % 2 == next_char % 2):
                # Swap and check if the new string is smaller
                # Create a temporary list to simulate the swap
                temp = chars.copy()
                temp[i], temp[i+1] = temp[i+1], temp[i]
                new_s = ''.join(temp)
                if new_s < s:
                    return new_s
        
        # If no swap leads to a smaller string, return the original
        return s