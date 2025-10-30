class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        # Define a function to perform the transformation on a single character
        def transform_char(char):
            if char == 'z':
                return "ab"
            else:
                return chr(ord(char) + 1)
        
        # Perform the transformations on the input string
        for _ in range(t):
            s = ''.join(transform_char(char) for char in s)
        
        # Return the length of the transformed string modulo 10^9 + 7
        return len(s) % (10**9 + 7)