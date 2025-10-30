class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def dfs(num, limit):
            if num > limit:
                return 0
            
            count = 1 if num >= int(low) else 0
            last_digit = num % 10
            
            if last_digit > 0:
                count += dfs(num * 10 + last_digit - 1, limit)
            if last_digit < 9:
                count += dfs(num * 10 + last_digit + 1, limit)
            
            return count % MOD
        
        def count_stepping_numbers(upper):
            if int(upper) < 10:
                return min(int(upper) - int(low) + 1, 10 - int(low))
            
            count = min(9, int(upper) - int(low) + 1) if int(low) < 10 else 0
            
            for i in range(1, 10):
                count += dfs(i, int(upper))
                count %= MOD
            
            return count
        
        return (count_stepping_numbers(high) - count_stepping_numbers(str(int(low) - 1)) + MOD) % MOD