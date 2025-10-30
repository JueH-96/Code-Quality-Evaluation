class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        count = 0
        sum_val = 0
        num = 1
        seen = set()
        
        while count < n:
            if target - num not in seen:
                sum_val = (sum_val + num) % MOD
                seen.add(num)
                count += 1
            num += 1
        
        return sum_val