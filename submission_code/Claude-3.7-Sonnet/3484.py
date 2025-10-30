class Solution:
    def getSmallestString(self, s: str) -> str:
        best = s
        for i in range(len(s) - 1):
            # Check if adjacent digits have the same parity
            if int(s[i]) % 2 == int(s[i + 1]) % 2:
                # Swap the digits
                swapped = s[:i] + s[i+1] + s[i] + s[i+2:]
                
                if swapped < best:
                    best = swapped
        
        return best