from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # A valid `original` array exists if and only if the bitwise XOR sum
        # of all elements in `derived` is 0. This is because when we sum
        # all the defining equations using XOR, each `original[i]` term
        # appears twice, canceling itself out (since x ^ x = 0).
        #
        # For a binary array (containing only 0s and 1s), the XOR sum is 0
        # if and only if there is an even number of 1s.
        # The number of 1s in the array is simply `sum(derived)`.
        #
        # Therefore, the condition simplifies to checking if `sum(derived)` is even.
        
        return sum(derived) % 2 == 0