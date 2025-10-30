class Solution:
    def waysToReachStair(self, k: int) -> int:
        def binomial(n, k):
            if k < 0 or k > n:
                return 0
            res = 1
            for i in range(1, k + 1):
                res = res * (n - i + 1) // i
            return res
        
        ans = 0
        for u in range(61):  # Loop from u=0 to u=60 to cover all possible u where 2^u >= k for k <= 10^9
            pow_u = 1 << u
            if pow_u >= k:
                d = pow_u - k
                ans += binomial(u + 1, d)
        return ans