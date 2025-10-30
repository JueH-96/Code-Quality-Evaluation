class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = 10**9 + 7
        
        # We first observe that:
        # 1. There are x^n ways to assign n performers to x stages.
        # 2. For any assignment, let k be the number of stages (bands) that actually got at least one performer.
        #    For each nonempty stage, the jury awards the band a score from 1 to y in y ways.
        #    Thus for that assignment the band scoring can happen in y^k ways.
        #
        # We can count the total number of events by summing over the number k of nonempty stages:
        #   Total = sum_{k=1}^{min(n, x)} ( ways_to_choose_nonempty_stages and assign performers
        #                                    such that exactly k stages are used * y^k  )
        #
        # To count the number of performer assignments that yield exactly k nonempty stages:
        #   a. Choose which k stages (from x) are nonempty: C(x, k)
        #   b. Partition the n performers into k nonempty groups. The number of ways is given by the Stirling
        #      numbers of the second kind S(n, k).
        #   c. Since the k chosen stages are labeled (i.e. order matters), multiply by k!.
        #
        # Thus, for a fixed k, the assignments count is: C(x, k) * S(n, k) * k!
        # And the overall term including band scores is: C(x, k) * S(n, k) * k! * y^k.
        #
        # Finally, we sum for k = 1 to min(n, x) and take the answer modulo mod.
        
        # Precompute S(n,k) using a 1D DP (Stirling numbers of the second kind):
        # Recurrence: S(n, k) = S(n-1, k-1) + k * S(n-1, k) for n>=1 & 1<=k<=n, with S(0,0)=1.
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            # Update in reverse to avoid overwriting needed data.
            for k in range(i, 0, -1):
                dp[k] = (dp[k - 1] + k * dp[k]) % mod
            dp[0] = 0  # S(n, 0)=0 for n>=1
        
        # Precompute factorials and inverse factorials up to max(x, n)
        m = max(x, n) + 1
        fact = [1] * (m)
        invfact = [1] * (m)
        for i in range(1, m):
            fact[i] = fact[i - 1] * i % mod
        invfact[m - 1] = pow(fact[m - 1], mod - 2, mod)
        for i in range(m - 2, -1, -1):
            invfact[i] = invfact[i + 1] * (i + 1) % mod

        def nCk(n_val, k_val):
            if k_val < 0 or k_val > n_val:
                return 0
            return fact[n_val] * invfact[k_val] % mod * invfact[n_val - k_val] % mod
        
        res = 0
        # k (number of nonempty stages) goes from 1 to min(n, x)
        for k in range(1, min(n, x) + 1):
            # Number of ways to choose k stages that are nonempty
            ways_stages = nCk(x, k)
            # The number of ways to partition n performers into k nonempty groups is dp[k]
            # Multiply by k! since the k groups can be assigned to the k chosen stages in k! ways.
            ways_assignment = ways_stages * dp[k] % mod * fact[k] % mod
            # For each band (nonempty stage), there are y possible scores, so multiply by y^k.
            ways_scores = pow(y, k, mod)
            term = ways_assignment * ways_scores % mod
            res = (res + term) % mod
        
        return res

# The following part is for local testing and input/output handling.
def solve():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    x = int(data[1])
    y = int(data[2])
    sol = Solution()
    result = sol.numberOfWays(n, x, y)
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    solve()