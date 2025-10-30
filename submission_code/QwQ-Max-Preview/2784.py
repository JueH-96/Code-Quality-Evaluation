class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        if n == 0:
            return 0
        
        # Initialize the prefix sum array
        s = [0] * n
        
        # Compute the prefix sums using dynamic programming
        for i in range(1, n):
            s[i] = (nums_sorted[i-1] + 2 * s[i-1]) % mod
        
        total = 0
        for i in range(n):
            current_sum = (s[i] + nums_sorted[i]) % mod
            contribution = (pow(nums_sorted[i], 2, mod) * current_sum) % mod
            total = (total + contribution) % mod
        
        return total % mod