from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        Checks if a valid binary array 'original' exists for a given 'derived' array.
        A valid 'original' array 'o' of length n generates a 'derived' array 'd'
        of length n such that:
        d[i] = o[i] ^ o[i+1] for 0 <= i < n-1
        d[n-1] = o[n-1] ^ o[0]

        We can XOR all the derived equations together:
        d[0] ^ d[1] ^ ... ^ d[n-2] ^ d[n-1]
        = (o[0] ^ o[1]) ^ (o[1] ^ o[2]) ^ ... ^ (o[n-2] ^ o[n-1]) ^ (o[n-1] ^ o[0])

        Using the properties of XOR (associativity and commutativity), and x ^ x = 0:
        The right side simplifies to:
        (o[0] ^ o[0]) ^ (o[1] ^ o[1]) ^ ... ^ (o[n-1] ^ o[n-1])
        = 0 ^ 0 ^ ... ^ 0
        = 0

        Thus, a necessary condition for a valid 'original' array to exist is that
        the XOR sum of all elements in the 'derived' array must be 0.
        d[0] ^ d[1] ^ ... ^ d[n-1] = 0

        This condition is also sufficient. If the XOR sum is 0, we can construct
        a valid 'original' array. For example, we can set o[0] = 0.
        Then, we can determine o[i] for i > 0 using the relation:
        o[i+1] = o[i] ^ d[i] for 0 <= i < n-1.
        This sequence satisfies the first n-1 derived equations by construction.
        Since the total XOR sum d[0]^...^d[n-1] = 0, the last equation
        d[n-1] = o[n-1] ^ o[0] will also be satisfied. All o[i] will be binary
        since we start with a binary value and only perform XOR with binary values.

        Therefore, a valid binary array 'original' exists if and only if the
        XOR sum of all elements in 'derived' is 0.
        """
        
        # Calculate the XOR sum of all elements in the derived array
        xor_sum = 0
        for val in derived:
            # XOR operation
            xor_sum ^= val
            
        # Check if the final XOR sum is 0
        return xor_sum == 0