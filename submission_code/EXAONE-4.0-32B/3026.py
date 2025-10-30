class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        mod = 10**9 + 7
        k = (target - 1) // 2
        if n <= k:
            return n * (n + 1) // 2 % mod
        
        total_sum = k * (k + 1) // 2
        count = k
        
        if target % 2 == 0:
            if count < n:
                total_sum += target // 2
                count += 1
        
        if count < n:
            remaining = n - count
            total_sum += remaining * (2 * target + remaining - 1) // 2
        
        return total_sum % mod