class Solution:
    def getSmallestString(self, s: str) -> str:
        best_s = s  # Start with the original string as the best candidate
        
        # Try swapping each pair of adjacent digits with the same parity exactly once
        for i in range(len(s) - 1):
            if (int(s[i]) % 2) == (int(s[i+1]) % 2):
                # Create a new string with the swap applied
                swapped = list(s)
                swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
                swapped_str = ''.join(swapped)
                
                # Update best string if this new swap results in a smaller string
                if swapped_str < best_s:
                    best_s = swapped_str
        
        return best_s