class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        # The problem transformation is equivalent to performing k cumulative sums.
        # It turns out that after k seconds, a[i] = C(i + k, k)
        # Therefore, a[n-1] = C(n + k - 1, k)
        # We compute C(n+k-1, k) = (n * (n+1) * ... * (n+k-1)) / k! modulo MOD
        
        result = 1
        for i in range(1, k + 1):
            result = result * (n + i - 1) % MOD
            # Multiply by modular inverse of i (since MOD is prime)
            result = result * pow(i, MOD - 2, MOD) % MOD
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.valueAfterKSeconds(4, 5))  # Expected output: 56
    print(sol.valueAfterKSeconds(5, 3))  # Expected output: 35