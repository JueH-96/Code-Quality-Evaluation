class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Precompute the length after each transformation
        # We need to find the length after t transformations
        # Since each transformation can be represented as a linear transformation on the counts of 'z's and other characters
        # We can model the growth of the string length using matrix exponentiation or dynamic programming
        
        # Let's define:
        # a = number of non-'z' characters
        # b = number of 'z' characters
        # After one transformation:
        # a_new = a (each non-'z' becomes another non-'z')
        # b_new = a (each 'z' becomes 'ab', which contributes 1 to a and 1 to b)
        # Wait, no. Each 'z' becomes 'ab', which is 2 characters: 'a' and 'b'
        # So, the new a is a (from non-'z') + b (from 'z' -> 'a')
        # The new b is b (from 'z' -> 'b')
        # Wait, no. Each 'z' becomes 'ab', which is 1 'a' and 1 'b'
        # So, the new a is a (from non-'z') + b (from 'z' -> 'a')
        # The new b is b (from 'z' -> 'b')
        # So, the transformation is:
        # a_new = a + b
        # b_new = b
        
        # So, the transformation matrix is:
        # [1 1]
        # [0 1]
        
        # To find the length after t transformations, we need to compute:
        # [a_t, b_t] = [a_0, b_0] * [1 1; 0 1]^t
        
        # The length of the string after t transformations is a_t + 2 * b_t
        
        # First, compute a_0 and b_0
        a = 0
        b = 0
        for char in s:
            if char == 'z':
                b += 1
            else:
                a += 1
        
        # Now, we need to compute [a_t, b_t] = [a, b] * [1 1; 0 1]^t
        # Since [1 1; 0 1]^t = [1 t; 0 1], we can directly compute:
        a_t = a + b * t
        b_t = b
        
        # The length is a_t + 2 * b_t
        length = a_t + 2 * b_t
        return length % MOD