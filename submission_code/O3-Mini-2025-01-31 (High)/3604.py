class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = 10**9 + 7

        # We want to count the total number of events.
        # Each event is defined by:
        #   • Assigning each of the n performers to one of x stages.
        #   • For each nonempty stage (i.e. a band), awarding it a score from 1 to y.
        #
        # Observe that any assignment gives a certain number j (1 ≤ j ≤ min(n, x)) of nonempty stages.
        # The number of assignments with exactly j nonempty stages is:
        #      comb(x, j) * (number of surjections from n performers to j stages)
        #   and the number of surjections is: j! * S(n, j),
        # where S(n, j) is the Stirling number of the second kind.
        # Then, for every such assignment, the jury can assign scores in y^j ways.
        #
        # So the overall answer is:
        #     Σ (j=1 to min(n, x)) [ comb(x, j) * j! * S(n, j) * (y^j) ]
        # We compute this modulo 10^9+7.
        
        # Step 1. Compute Stirling numbers of the second kind S(n,j) for j = 0,...,n.
        # We use the recurrence: S(n, j) = S(n-1, j-1) + j * S(n-1, j)
        # with base S(0,0)=1 and S(n,0)=0 for n>=1.
        dp = [0] * (n + 1)
        dp[0] = 1  # S(0,0)=1
        for i in range(1, n + 1):
            # Update dp in reverse order so that dp[j-1] of the previous iteration remains available.
            for j in range(i, 0, -1):
                dp[j] = (dp[j - 1] + j * dp[j]) % mod
            dp[0] = 0  # For i>=1, S(i, 0) is 0.
        
        # After the loop, dp[j] holds S(n, j) for j = 0,...,n.
        
        # Step 2. Precompute factorials and inverse factorials up to max(x, n).
        max_val = max(x, n)
        fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = fact[i - 1] * i % mod
            
        invfact = [1] * (max_val + 1)
        invfact[max_val] = pow(fact[max_val], mod - 2, mod)  # Using Fermat's little theorem.
        for i in range(max_val, 0, -1):
            invfact[i - 1] = invfact[i] * i % mod
        
        # Step 3. Sum the contributions for each possible number j of nonempty stages.
        # Note that comb(x, j) * j! = fact[x] / fact[x - j]  (computed modulo mod)
        total = 0
        limit = min(n, x)
        for j in range(1, limit + 1):
            # p represents the ordered selection of j stages out of x
            p = fact[x] * invfact[x - j] % mod
            # dp[j] is S(n, j) so j! * S(n, j) is the number of surjections from n performers to j stages.
            term = p * dp[j] % mod
            term = term * pow(y, j, mod) % mod  # Each band (nonempty stage) gets one of y scores.
            total = (total + term) % mod
        return total

# Test cases provided in the question:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.numberOfWays(1, 2, 3))   # Expected output: 6
    
    # Example 2:
    print(sol.numberOfWays(5, 2, 1))   # Expected output: 32
    
    # Example 3:
    print(sol.numberOfWays(3, 3, 4))   # Expected output: 684