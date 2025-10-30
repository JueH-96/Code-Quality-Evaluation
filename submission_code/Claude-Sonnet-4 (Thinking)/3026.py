class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        count1 = target // 2
        
        if n <= count1:
            # Take the first n numbers from [1, 2, ..., count1]
            return (n * (n + 1) // 2) % MOD
        else:
            # Take all count1 numbers from [1, 2, ..., count1]
            # and (n - count1) numbers from [target, target + 1, ...]
            sum1 = count1 * (count1 + 1) // 2
            remaining = n - count1
            sum2 = remaining * target + remaining * (remaining - 1) // 2
            return (sum1 + sum2) % MOD