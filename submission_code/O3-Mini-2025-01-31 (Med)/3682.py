MOD = 10**9 + 7

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        # Explanation:
        # We want arrays of length n with entries in [1, m] and exactly k indices i (1 <= i < n)
        # such that arr[i-1] == arr[i].
        #
        # Notice that for positions 2 through n (i.e. the (n-1) "gaps") we make a decision:
        # either we "stay" (arr[i] is same as arr[i-1], contributing 1 to the count k)
        # or we "change" (arr[i] must be any value different from arr[i-1], contributing 0).
        #
        # Exactly k of these (n-1) positions must be the "stay" option, and the remaining (n-1-k)
        # positions must be the "change" option.
        #
        # For the first element, we have m choices.
        # For each "stay" decision, we have exactly 1 possibility (to copy the previous element).
        # For each "change" decision, we have (m-1) possibilities (all elements except the previous one).
        #
        # Moreover, we must choose which k indices (among the n-1 available spots) will have the "stay" decision.
        # This can be done in C(n-1, k) ways.
        #
        # Therefore, the total number of good arrays is:
        #   m * [C(n-1, k)] * (m-1)^(n-1-k) modulo MOD.
        #
        # Compute combination C(n-1, k) using an iterative approach to work mod MOD.
        #
        # If m is 0 (should not happen because m>=1), simply return 0.
        
        if m == 0:
            return 0
        
        # Compute C(n-1, k) mod MOD iteratively:
        c = 1
        # Using the formula: C(n-1, k) = ( (n-1) * (n-2) * ... * (n-k) ) / (1 * 2 * ... * k).
        for i in range(1, k+1):
            c = c * (n - i) % MOD
            c = c * pow(i, MOD-2, MOD) % MOD  # Modular inverse of i modulo MOD
        
        # Compute (m-1)^(n-1-k) modulo MOD:
        part = pow(m-1, n-1-k, MOD)
        
        # Combine the parts together:
        ans = (m * c) % MOD
        ans = (ans * part) % MOD
        
        return ans

# Below is a simple test routine.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.countGoodArrays(3, 2, 1))  # Expected output: 4
    # Example 2:
    print(sol.countGoodArrays(4, 2, 2))  # Expected output: 6
    # Example 3:
    print(sol.countGoodArrays(5, 2, 0))  # Expected output: 2