class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Explanation:
        # In one operation we subtract (2^i + num2) from num1. If we do k operations (with possibly
        # different choices of i) the total subtracted is
        #      (2^(i1) + num2) + (2^(i2) + num2) + ... + (2^(ik) + num2)
        #    = (sum of k (allowed) powers‐of‐2) + k*num2.
        #
        # To obtain zero we must have:
        #      num1 – k*num2 = S,
        # where S is the sum of k powers–of–2. (Allowed powers are 2^i with i in [0,60].)
        #
        # Notice that if you use k operations then:
        #   • each term is at least 1 (since 2^0 = 1) so S >= k,
        #   • and each term is at most 2^60 so S <= k * (2^60).
        #
        # Moreover, recall that every positive integer S has a unique binary expansion; however if we
        # are allowed to “split” a power–of–2 (that is, choose the same power more than once) then S can be
        # represented as a sum of exactly k powers–of–2 if and only if its “binary‐representation count”
        # (i.e. its popcount) is at most k. (Because if popcount(S) = r then at least r summands are needed,
        # and if r ≤ k one can “split” some nonzero bits into several smaller powers of 2 in order to use exactly k parts.)
        #
        # Thus the necessary conditions to be able to “reach zero” in exactly k operations are that if we let
        #    S = num1 – k*num2   (note that if num2 < 0 then S = num1 + |num2|*k)
        # we must have:
        #   (1) S >= 0,
        #   (2) S >= k    (since with k summands, the minimal sum is k*1 = k),
        #   (3) S <= k*(2^60)  (since every summand is at most 2^60),
        #   (4) popcount(S) ≤ k.
        #
        # In addition, if num2 == 0 then S = num1 and the answer is simply the number of 1’s in the binary
        # representation of num1.
        #
        # A short analysis shows that if a solution exists the minimal k does not exceed 61.
        # Hence we try all k from 1 to 61.
        
        if num2 == 0:
            return num1.bit_count()
        
        for k in range(1, 62):
            if num2 > 0:
                # When num2 is positive S decreases as k increases.
                S = num1 - k * num2
                if S < 0:
                    # Further increases in k will give even smaller (negative) S.
                    break
            else:
                # When num2 is negative, note that S = num1 - k*num2 = num1 + |num2|*k.
                S = num1 - k * num2
            
            # We must have S >= k (since k ones sum to k)
            if S < k:
                continue
            # Also S must be at most k*(2^60) because every operation subtracts at most 2^60 in its power‐term.
            if S > k * (1 << 60):
                continue
            
            # For numbers S less than 2^60 the minimum number of summands needed is S.bit_count().
            # (If S is larger we “split” any term above 2^60; that minimal count equals
            #    (S >> 60) + popcount(S % (2^60)) ).
            if S >= (1 << 60):
                required = (S >> 60) + ((S & ((1 << 60) - 1)).bit_count())
            else:
                required = S.bit_count()
            
            # If S can be represented with exactly k powers–of–2 (i.e. if the minimal number 
            # required is ≤ k) then a solution is possible.
            if required <= k:
                return k

        return -1

# --- Testing the solution ---
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.makeTheIntegerZero(3, -2))  # Expected output: 3
    # Example 2:
    print(sol.makeTheIntegerZero(5, 7))   # Expected output: -1

    # Additional testing: (you can add more tests if needed)
    # For instance, when num2 is positive:
    print(sol.makeTheIntegerZero(1000000000, 1))
    
# Note: For submission the testing code at the bottom isn’t necessary.