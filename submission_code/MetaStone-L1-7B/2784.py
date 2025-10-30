class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        a_sorted = sorted(nums)
        n = len(a_sorted)
        a_squared = [x * x % MOD for x in a_sorted]
        S = [0] * (n + 1)  # S[j] is the sum for k >= j of 2^{k-j} * a_k^2
        
        for j in range(n - 1, -1, -1):
            S[j] = (a_squared[j] + 2 * S[j + 1]) % MOD
        
        total = 0
        for j in range(n):
            current = a_sorted[j]
            term = current * (a_squared[j] + S[j + 1]) % MOD
            total = (total + term) % MOD
        
        return total % MOD