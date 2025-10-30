from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        Determines if a valid binary array 'original' exists that could have formed 'derived'.

        Args:
            derived: A 0-indexed array of 0s and 1s.

        Returns:
            True if such an array exists, False otherwise.
        
        Let n be the length of the derived array. The rules for forming 'derived' from
        a binary array 'original' of length n are:
        1. derived[i] = original[i] ^ original[i+1] for 0 <= i < n-1
        2. derived[n-1] = original[n-1] ^ original[0]
        where ^ denotes the bitwise XOR operation.

        Consider the XOR sum of all elements in the 'derived' array:
        xor_sum_derived = derived[0] ^ derived[1] ^ ... ^ derived[n-1]
        
        Substituting the definitions of derived[i]:
        xor_sum_derived = (original[0] ^ original[1]) ^ 
                          (original[1] ^ original[2]) ^ 
                          ... ^
                          (original[n-2] ^ original[n-1]) ^ 
                          (original[n-1] ^ original[0])
                          
        Using the associative and commutative properties of the XOR operation, we can
        rearrange and group the terms involving the elements of the 'original' array:
        xor_sum_derived = (original[0] ^ original[0]) ^ 
                          (original[1] ^ original[1]) ^ 
                          ... ^
                          (original[n-1] ^ original[n-1])
                          
        We know that any value XORed with itself results in 0 (i.e., x ^ x = 0).
        Therefore, each pair (original[i] ^ original[i]) evaluates to 0:
        xor_sum_derived = 0 ^ 0 ^ ... ^ 0
        xor_sum_derived = 0
        
        This derivation shows that a necessary condition for a valid 'original' array
        to exist is that the XOR sum of all elements in the 'derived' array must be 0.
        
        This condition is also sufficient. If the XOR sum of 'derived' is 0, it can be
        shown that we can always construct at least one valid 'original' array. 
        For instance, we can assume a value for original[0] (either 0 or 1) and then
        iteratively determine the values of original[1], original[2], ..., original[n-1]
        using the first n-1 derivation rules:
        original[i+1] = derived[i] ^ original[i] for 0 <= i < n-1.
        The final rule, derived[n-1] = original[n-1] ^ original[0], serves as a
        consistency check. This check will pass if and only if the XOR sum of the 
        'derived' array is 0.

        Thus, the problem reduces to checking if the XOR sum of the elements in the
        'derived' array is equal to 0.
        """
        
        xor_sum = 0
        # Iterate through the derived array and compute the XOR sum of its elements.
        # The XOR operation (^) is equivalent to addition modulo 2 for binary numbers.
        for d in derived:
            xor_sum ^= d
            
        # If the XOR sum is 0, it means a valid 'original' array could exist.
        # If the XOR sum is 1, no such 'original' array can exist.
        # The function should return True if xor_sum is 0, and False otherwise.
        return xor_sum == 0