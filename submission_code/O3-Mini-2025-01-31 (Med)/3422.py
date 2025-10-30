MOD = 10**9 + 7

class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        # The problem can be solved by recognizing that after k seconds,
        # the value of a[n - 1] is equal to the binomial coefficient C((n - 1) + k, k)
        # This is because each step performs a prefix sum, which is equivalent to adding an extra row of Pascal's Triangle.
        
        # We therefore need to calculate C(n - 1 + k, k) modulo MOD.
        N = n - 1 + k  # Highest value for factorial calculation
        
        # Precompute factorials up to N
        fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        # Precompute inverse factorials (using Fermat's little theorem for mod p)
        invfact = [1] * (N + 1)
        invfact[N] = pow(fact[N], MOD - 2, MOD)
        for i in range(N, 0, -1):
            invfact[i - 1] = invfact[i] * i % MOD
        
        # Compute the binomial coefficient: C(n - 1 + k, k) = fact[n-1+k] / (fact[n-1] * fact[k])
        return fact[N] * invfact[n - 1] % MOD * invfact[k] % MOD

# Example test runs:
if __name__ == '__main__':
    sol = Solution()
    print(sol.valueAfterKSeconds(4, 5))  # Output should be 56
    print(sol.valueAfterKSeconds(5, 3))  # Output should be 35