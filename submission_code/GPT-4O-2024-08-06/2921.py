class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def is_stepping_number(num: str) -> bool:
            for i in range(1, len(num)):
                if abs(int(num[i]) - int(num[i - 1])) != 1:
                    return False
            return True
        
        def count_stepping_numbers_below(n: str) -> int:
            if n == "0":
                return 0
            
            length = len(n)
            count = 0
            
            # Count stepping numbers with fewer digits
            for l in range(1, length):
                for start_digit in range(1, 10):
                    count += count_stepping_numbers_of_length(l, start_digit)
                    count %= MOD
            
            # Count stepping numbers with the same number of digits
            for start_digit in range(1, int(n[0])):
                count += count_stepping_numbers_of_length(length, start_digit)
                count %= MOD
            
            # Count stepping numbers starting with the same first digit
            current_number = n[0]
            for i in range(1, length):
                next_digit = int(n[i])
                prev_digit = int(current_number[-1])
                
                if abs(prev_digit - next_digit) != 1:
                    break
                
                for d in range(0, next_digit):
                    if abs(prev_digit - d) == 1:
                        count += count_stepping_numbers_of_length(length - i, d)
                        count %= MOD
                
                current_number += n[i]
            
            if is_stepping_number(n):
                count += 1
                count %= MOD
            
            return count
        
        def count_stepping_numbers_of_length(l: int, start_digit: int) -> int:
            if l == 1:
                return 1
            
            dp = [[0] * 10 for _ in range(l)]
            dp[0][start_digit] = 1
            
            for i in range(1, l):
                for d in range(10):
                    if dp[i - 1][d] > 0:
                        if d > 0:
                            dp[i][d - 1] += dp[i - 1][d]
                            dp[i][d - 1] %= MOD
                        if d < 9:
                            dp[i][d + 1] += dp[i - 1][d]
                            dp[i][d + 1] %= MOD
            
            return sum(dp[l - 1]) % MOD
        
        low_int = int(low)
        high_int = int(high)
        
        count_high = count_stepping_numbers_below(high)
        count_low = count_stepping_numbers_below(str(low_int - 1))
        
        return (count_high - count_low + MOD) % MOD