class Solution:
    def getSmallestString(self, s: str) -> str:
        # Start by assuming the best (smallest) string is the original string
        best_str = s
        
        # Convert the string to a list of characters for easier swapping
        chars = list(s)
        n = len(chars)
        
        # Try swapping each adjacent pair that has the same parity exactly once
        for i in range(n - 1):
            # Check if the two adjacent digits have the same parity
            if (int(chars[i]) % 2) == (int(chars[i+1]) % 2):
                # Swap
                chars[i], chars[i+1] = chars[i+1], chars[i]
                candidate = "".join(chars)
                # Update the best_str if this new string is lexicographically smaller
                if candidate < best_str:
                    best_str = candidate
                # Swap back to restore original order
                chars[i], chars[i+1] = chars[i+1], chars[i]
        
        return best_str