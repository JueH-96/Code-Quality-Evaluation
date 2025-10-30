class Solution:
    def getSmallestString(self, s: str) -> str:
        # We can perform at most one swap of two adjacent digits with the same parity.
        # Our goal is to find the lexicographically smallest string after at most one such swap.
        
        # Convert the string to a list for easy swapping
        chars = list(s)
        best_str = s  # Start by assuming the original string is the smallest
        
        # Helper function to check if two digits have the same parity
        def same_parity(a, b):
            return (int(a) % 2) == (int(b) % 2)
        
        n = len(chars)
        for i in range(n - 1):
            # If adjacent digits have the same parity, try swapping them
            if same_parity(chars[i], chars[i + 1]):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                new_s = "".join(chars)
                
                # Update the best string if we get a smaller lexicographic ordering
                if new_s < best_str:
                    best_str = new_s
                
                # Swap back to restore original configuration
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        
        return best_str