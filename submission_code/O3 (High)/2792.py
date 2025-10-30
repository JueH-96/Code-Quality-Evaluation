from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        For the system of equations
            o[i] XOR o[(i+1) % n] = derived[i]   (0 ≤ i < n)
        the variables o[i] are solvable over GF(2) iff
            XOR_{i=0}^{n-1} derived[i] == 0.

        Proof sketch:
        • Choose any value for o[0] (0 or 1).
        • Recursively set  o[i+1] = o[i] XOR derived[i]  for i=0…n−2.
        • Consistency requires the value obtained for o[n] equals o[0]:
              o[n] = o[0] XOR (derived[0] ⊕ … ⊕ derived[n−1]) == o[0]
          which is true exactly when the XOR of all derived entries is 0.

        Hence the array `derived` is valid ⇔ its XOR is 0.
        """
        xor_sum = 0
        for v in derived:
            xor_sum ^= v
        return xor_sum == 0