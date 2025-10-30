MOD = 10**9+7

class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        # Explanation:
        # We want to count all binary arrays that have exactly 'zero' zeros and 'one' ones and,
        # importantly, no contiguous block (i.e. “run”) of 0’s or 1’s is longer than "limit".
        # (Any subarray of length > limit contained entirely in one run would lack the other digit.)
        #
        # Notice that if you "compress" any binary array (i.e. record each maximal run of identical
        # characters), you get an alternating sequence. Thus every valid array is uniquely determined
        # by (a) the starting digit and (b) the run‐lengths for zeros and ones.
        #
        # For example, consider an array starting with 0 and ending with 0:
        #   Pattern: 0-run, 1-run, 0-run, 1-run, …, 0-run.
        # Let x = number of runs of 0’s.
        # Then the number of runs of 1’s is x-1.
        # The array has “zero” many 0’s and “one” many 1’s. In other words, we are choosing
        # an ordered partition (composition) of zero into x parts and one into x-1 parts,
        # with the extra restriction that each part (i.e. run) is at most "limit" (and at least 1).
        #
        # Similarly, if an array starts with 0 and ends with 1, then the pattern is:
        #   0-run, 1-run, 0-run, 1-run, …, 0-run, 1-run.
        # Here, if we let x = number of 0‐runs then the number of 1‐runs is x.
        #
        # We similarly get two cases for arrays starting with 1.
        #
        # So we break the total count into four cases:
        #   Case 1: start0, end0: zeros partitioned into x parts and ones into x-1 parts (with x>=2).
        #   Case 2: start0, end1: zeros partitioned into x parts and ones into x parts (x>=1).
        #   Case 3: start1, end1: ones partitioned into x parts and zeros into x-1 parts (x>=2).
        #   Case 4: start1, end0: ones partitioned into x parts and zeros into x parts (x>=1).
        #
        # To count the number of compositions (ordered partitions) of an integer n into exactly r parts,
        # where each part is at least 1 and at most L, we use an inclusion–exclusion formula.
        #
        # Define the function:
        #    f(n, r, L) = number of ways to choose r positive integers a1, a2, …, ar with 1 <= ai <= L and a1+a2+...+ar = n.
        #
        # It is known that if n < r or n > r*L then f(n, r, L)=0. Otherwise:
        #    f(n, r, L) = sum_{j=0}^{floor((n-r)/L)} (-1)^j * C(r, j) * C(n - 1 - j*L, r - 1)
        #
        # We precompute the combinations needed using factorials (with mod MOD).
        
        # precompute factorials and their modular inverses (up to a safe bound)
        maxN = 500  # This is more than enough because zero, one <= 200.
        fact = [1]*(maxN+1)
        invfact = [1]*(maxN+1)
        for i in range(1, maxN+1):
            fact[i] = fact[i-1] * i % MOD
        invfact[maxN] = pow(fact[maxN], MOD-2, MOD)
        for i in range(maxN, 0, -1):
            invfact[i-1] = invfact[i] * i % MOD

        def nCr(n, r):
            if r < 0 or r > n:
                return 0
            return fact[n] * invfact[r] % MOD * invfact[n-r] % MOD
        
        # compositions(n, r, L): number of ways to write n as the sum of r parts in [1, L]
        def compositions(n, r, L):
            if n < r or n > r * L:
                return 0
            total = 0
            max_j = (n - r) // L
            for j in range(max_j+1):
                # sign = 1 if j is even and -1 if odd (using mod arithmetic, -1 is MOD-1)
                sign = 1 if j % 2 == 0 else (MOD - 1)
                term = nCr(r, j) * nCr(n - 1 - j * L, r - 1) % MOD
                total = (total + sign * term) % MOD
            return total % MOD

        res = 0
        
        # Case 1: Start with 0 and end with 0.
        # Pattern: 0,1,0,1,...,0.
        # Let x = number of 0-runs.
        # Then the number of 1-runs is x-1.
        # x must be at least 2 (since one must occur) and satisfy: x <= zero and (x-1) <= one.
        for x in range(2, min(zero, one+1) + 1):
            ways0 = compositions(zero, x, limit)
            ways1 = compositions(one, x-1, limit)
            res = (res + ways0 * ways1) % MOD

        # Case 2: Start with 0 and end with 1.
        # Pattern: 0,1,0,1,...,0,1.
        # Let x = number of 0-runs; then the number of 1-runs is also x.
        # We require: x >= 1, x <= zero, and x <= one.
        for x in range(1, min(zero, one) + 1):
            ways0 = compositions(zero, x, limit)
            ways1 = compositions(one, x, limit)
            res = (res + ways0 * ways1) % MOD
        
        # Case 3: Start with 1 and end with 1.
        # Pattern: 1,0,1,0,...,1.
        # Let x = number of 1-runs; then zeros are in x-1 runs.
        for x in range(2, min(one, zero+1) + 1):
            ways1 = compositions(one, x, limit)
            ways0 = compositions(zero, x-1, limit)
            res = (res + ways1 * ways0) % MOD
        
        # Case 4: Start with 1 and end with 0.
        # Pattern: 1,0,1,0,...,1,0.
        # Let x = number of 1-runs; then zeros are in x runs.
        for x in range(1, min(one, zero) + 1):
            ways1 = compositions(one, x, limit)
            ways0 = compositions(zero, x, limit)
            res = (res + ways1 * ways0) % MOD
        
        return res % MOD


# The following code is for local testing.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.numberOfStableArrays(1, 1, 2))  # Expected output 2
    # Example 2:
    print(sol.numberOfStableArrays(1, 2, 1))  # Expected output 1
    # Example 3:
    print(sol.numberOfStableArrays(3, 3, 2))  # Expected output 14