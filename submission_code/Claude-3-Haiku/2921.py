class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10 ** 9 + 7
        
        def is_stepping(num):
            if num < 10:
                return True
            
            prev_digit = num % 10
            num //= 10
            
            while num:
                curr_digit = num % 10
                if abs(curr_digit - prev_digit) != 1:
                    return False
                prev_digit = curr_digit
                num //= 10
            
            return True
        
        def dfs(num):
            if num > int(high):
                return 0
            
            count = 0
            if int(low) <= num <= int(high) and is_stepping(num):
                count += 1
            
            for digit in range(10):
                if digit == 0 and num == 0:
                    continue
                count += dfs(num * 10 + digit)
            
            return count % MOD
        
        return dfs(0)