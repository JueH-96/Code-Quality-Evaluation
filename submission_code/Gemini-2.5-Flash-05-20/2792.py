from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        Determines if a valid binary array original exists that could have formed the given derived array.

        The problem states that for a 0-indexed array 'derived' of length n,
        it is formed from a binary array 'original' of length n as follows:
        - For i in [0, n - 2]: derived[i] = original[i] ⊕ original[i + 1]
        - For i = n - 1: derived[n - 1] = original[n - 1] ⊕ original[0]

        We can sum (XOR) all the equations:
        derived[0] ⊕ derived[1] ⊕ ... ⊕ derived[n-1]
        = (original[0] ⊕ original[1]) ⊕
          (original[1] ⊕ original[2]) ⊕
          ... ⊕
          (original[n-2] ⊕ original[n-1]) ⊕
          (original[n-1] ⊕ original[0])

        Due to the associative and commutative properties of XOR (and x ⊕ x = 0):
        The right side simplifies to:
        (original[0] ⊕ original[0]) ⊕ (original[1] ⊕ original[1]) ⊕ ... ⊕ (original[n-1] ⊕ original[n-1])
        = 0 ⊕ 0 ⊕ ... ⊕ 0 (n times)
        = 0

        Therefore, if a valid 'original' array exists, the XOR sum of all elements
        in the 'derived' array must be 0. This is a necessary condition.

        It can also be shown that this condition is sufficient. If the XOR sum of
        'derived' elements is 0, we can construct a valid 'original' array. For example,
        by assuming original[0] = 0, we can uniquely determine all other original[i]
        values using the forward propagation: original[i+1] = derived[i] ⊕ original[i].
        The final wrap-around condition (derived[n-1] = original[n-1] ⊕ original[0])
        will automatically be satisfied if the total XOR sum of 'derived' is 0.

        So, the solution boils down to checking if the XOR sum of all elements in 'derived' is 0.
        """

        xor_sum = 0
        for val in derived:
            xor_sum ^= val
        
        return xor_sum == 0