from typing import List
# No need for functools or operator if using a simple loop.

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        Determines whether there exists a valid binary array original that could have formed derived.

        A 0-indexed array derived with length n is derived by computing the bitwise XOR (⊕)
        of adjacent values in a binary array original of length n.
        Specifically, for each index i in the range [0, n - 1]:
        - If i = n - 1, then derived[i] = original[i] ⊕ original[0].
        - Otherwise, derived[i] = original[i] ⊕ original[i + 1].

        Args:
          derived: The derived binary array.

        Returns:
          True if such an original array exists, false otherwise.
        """

        # As derived in the explanation, a valid `original` array exists if and only if
        # the XOR sum of all elements in the `derived` array is 0.
        
        # Constraints: n = len(derived), 1 <= n <= 10^5.
        # derived[i] is 0 or 1.
        # Thus, `derived` is never empty.

        current_xor_sum = 0
        for x_val in derived:
            current_xor_sum ^= x_val  # Apply bitwise XOR
        
        # If current_xor_sum is 0, a valid original array exists.
        # Otherwise, it does not.
        return current_xor_sum == 0