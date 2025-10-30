class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def is_stepping(num_str):
            if len(num_str) == 1:
                return True
            for i in range(len(num_str) - 1):
                if abs(int(num_str[i]) - int(num_str[i+1])) != 1:
                    return False
            return True

        def count_stepping_numbers_less_than_or_equal_to(num_str):
            n = len(num_str)
            dp = {}

            def solve(idx, is_tight, prev_digit):
                if idx == n:
                    return 1
                
                if (idx, is_tight, prev_digit) in dp:
                    return dp[(idx, is_tight, prev_digit)]
                
                ans = 0
                upper_bound = int(num_str[idx]) if is_tight else 9
                
                for digit in range(upper_bound + 1):
                    if idx == 0 and digit == 0:
                        continue
                    if idx > 0 and abs(digit - prev_digit) != 1:
                        continue
                    
                    ans = (ans + solve(idx + 1, is_tight and (digit == upper_bound), digit)) % MOD
                
                dp[(idx, is_tight, prev_digit)] = ans
                return ans
            
            return solve(0, True, -1)

        
        count_high = count_stepping_numbers_less_than_or_equal_to(high)
        count_low = count_stepping_numbers_less_than_or_equal_to(str(int(low) - 1)) if int(low) > 1 else 0
        
        return (count_high - count_low + MOD) % MOD