MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        # We want to count the number of ways for the event:
        # First: assign each performer a stage (1..x). Then, among the stages that get assigned at least
        # one performer (non-empty stages), each band (non-empty stage) gets a score in [1, y].
        #
        # Let m be the number of non-empty stages. Then:
        #   - the number of ways to assign performers so that exactly m distinct stages are used
        #     is: C(x, m) * (number of surjections from n items to m labels)
        #   - number of surjections = m! * S(n, m), where S(n, m) is the Stirling number of the second kind.
        #   - each band gets a score, so additional factor y^(m).
        #
        # Thus the answer is:
        #   Sum_{m=1 to min(n, x)} [ C(x, m) * m! * S(n, m) * y^(m) ]  mod MOD.
        
        # Precompute Stirling numbers S(n, m) for n performers and m bands for 0 <= m <= n.
        # We only need S(n, m) for 1 <= m <= min(n, x).
        # S(n, m) recurrence:
        #   S(n, m) = m * S(n-1, m) + S(n-1, m-1)
        # with S(0,0)=1 and S(n, 0)=0 for n>0.
        
        # We'll compute a 2D DP table dp[i][j] = S(i, j)
        maxN = n
        dp = [[0] * (maxN + 1) for _ in range(maxN + 1)]
        dp[0][0] = 1
        for i in range(1, maxN + 1):
            for j in range(1, i + 1):
                dp[i][j] = (j * dp[i-1][j] + dp[i-1][j-1]) % MOD
        
        # We also need factorials and inverse factorials for computing combinations and m! efficiently.
        # We'll compute up to max(x, n).
        maxVal = max(x, n)
        fact = [1] * (maxVal + 1)
        invfact = [1] * (maxVal + 1)
        for i in range(2, maxVal + 1):
            fact[i] = fact[i-1] * i % MOD
        invfact[maxVal] = pow(fact[maxVal], MOD-2, MOD)
        for i in range(maxVal, 0, -1):
            invfact[i-1] = invfact[i] * i % MOD
        
        # Combination function: C(N, r) = fact[N] / (fact[r]*fact[N-r])
        def comb(N, r):
            if r < 0 or r > N:
                return 0
            return fact[N] * invfact[r] % MOD * invfact[N-r] % MOD
        
        ans = 0
        # m can be from 1 to min(n, x)
        upper = min(n, x)
        for m in range(1, upper + 1):
            ways_assign = comb(x, m)  # ways to choose which m stages are non-empty.
            ways_surj = fact[m] * dp[n][m] % MOD  # m! * S(n, m)
            ways_score = pow(y, m, MOD)  # assign score to each nonempty stage
            term = ways_assign * ways_surj % MOD * ways_score % MOD
            ans = (ans + term) % MOD
        return ans