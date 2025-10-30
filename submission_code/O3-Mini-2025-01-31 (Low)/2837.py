class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We need to subtract (2^i + num2) in each operation.
        # In k operations, if we choose indices i1, i2, …, ik, the total subtracted amount is:
        #   (2^(i1) + num2) + ... + (2^(ik) + num2) = (2^(i1)+...+2^(ik)) + k*num2.
        # To make num1 become zero, we need:
        #    num1 - [sum(2^(ij)) + k * num2] = 0, or equivalently,
        #    sum(2^(ij)) = num1 - k * num2.
        # Let S = num1 - k * num2.
        # In each operation, we add one power-of-2 (from 2^0 up to 2^60), and note that we are allowed
        # to repeat the same power. Therefore, S must be expressible as a sum of k powers of 2.
        # A necessary and sufficient condition for an integer S to be expressed as
        # the sum of k (not necessarily distinct) powers of 2 is:
        #    - S must be non-negative.
        #    - The number of 1-bits in the binary representation of S (i.e. popcount(S)) 
        #      is at most k, because each 1-bit represents one distinct power-of-2 term.
        #    - Also, since each term is at least 1, we must have k <= S.
        #
        # In addition, note that each operation uses a power i in the range [0,60]. When summing up k terms,
        # the largest possible sum with all terms being at most 2^60 is k * (2^60).
        # Thus, we also require:
        #      S <= k * (2^60).
        #
        # Now, our task reduces to finding the minimum k >= 1 such that:
        #    S = num1 - k * num2 is nonnegative,
        #    popcount(S) <= k <= S, and
        #    S <= k * (2^60).
        #
        # We iterate over possible k and return the first valid one. If none is found,
        # return -1.
        
        # A helper function for popcount:
        def popcount(x: int) -> int:
            return bin(x).count("1")
        
        # Since num1 is at most 1e9 and num2 is in [-1e9, 1e9], 
        # it turns out that the required k (if it exists) is not very large.
        # We iterate over k from 1 up to a reasonable limit.
        for k in range(1, 10**6):
            S = num1 - k * num2
            if S < k:  # Cannot break S into k powers (each operation contributes at least 1)
                continue
            # Also, ensure that S can be formed using k powers that are at most 2^60.
            if S > k * (1 << 60):
                continue
            # Check if S can be decomposed into k powers-of-2.
            # Since any sum representation as powers-of-2 must include at least popcount(S) ones,
            # it is necessary that popcount(S) <= k.
            if popcount(S) <= k:
                return k
        return -1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.makeTheIntegerZero(3, -2))  # Expected output: 3
    # Example 2:
    print(sol.makeTheIntegerZero(5, 7))   # Expected output: -1