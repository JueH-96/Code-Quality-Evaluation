class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Initial length of the string
        length = len(s)
        
        # For each transformation
        for _ in range(t):
            # Calculate the new length after this transformation
            new_length = 0
            for char in s:
                if char == 'z':
                    new_length += 2  # 'z' becomes "ab"
                else:
                    new_length += 1  # Any other character becomes the next character
            
            # Update the length for the next transformation
            length = new_length
            
            # If the length exceeds the modulo, we can take it modulo to avoid overflow
            length %= MOD
        
        return length