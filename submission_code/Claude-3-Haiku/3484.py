class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        digits = [int(c) for c in s]
        
        # Find the first pair of adjacent digits with the same parity
        i = 0
        while i < n - 1 and (digits[i] % 2 != digits[i+1] % 2):
            i += 1
        
        # If no such pair is found, return the original string
        if i == n - 1:
            return s
        
        # Swap the adjacent digits with the same parity
        digits[i], digits[i+1] = digits[i+1], digits[i]
        
        # Convert the digits back to a string and return it
        return ''.join(str(d) for d in digits)