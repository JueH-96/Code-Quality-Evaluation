class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 1000000007
        
        def is_valid(num):
            s = str(num)
            for i in range(len(s)-1):
                if abs(int(s[i]) - int(s[i+1])) != 1:
                    return False
            return True
            
        def count_stepping_numbers(n):
            if len(n) == 1:
                return int(n)
                
            dp = [[0] * 10 for _ in range(len(n))]
            
            # Initialize first digit (can't start with 0)
            upper = int(n[0])
            for d in range(1, upper+1):
                dp[0][d] = 1
                
            # Fill dp array
            for pos in range(1, len(n)):
                curr_digit = int(n[pos])
                
                for prev in range(10):
                    if dp[pos-1][prev] == 0:
                        continue
                        
                    for d in range(10):
                        if abs(prev - d) == 1:
                            if pos == len(n)-1 and d > curr_digit:
                                continue
                            dp[pos][d] += dp[pos-1][prev]
                            
            return sum(dp[-1]) % MOD
            
        def get_stepping_numbers(num):
            if len(num) == 1:
                return sum(1 for i in range(1, int(num)+1))
                
            total = count_stepping_numbers(num)
            
            # Add stepping numbers with fewer digits
            for length in range(1, len(num)):
                for first in range(1, 10):
                    dp = [[0] * 10 for _ in range(length)]
                    dp[0][first] = 1
                    
                    for pos in range(1, length):
                        for prev in range(10):
                            if dp[pos-1][prev] == 0:
                                continue
                            for d in range(10):
                                if abs(prev - d) == 1:
                                    dp[pos][d] += dp[pos-1][prev]
                                    
                    total = (total + sum(dp[-1])) % MOD
                    
            return total
            
        # Convert strings to lists of digits
        low_int = int(low)
        high_int = int(high)
        
        # Handle edge cases
        if low_int == high_int:
            return 1 if is_valid(low_int) else 0
            
        # Get count for numbers up to high and subtract count for numbers up to low-1
        result = (get_stepping_numbers(high) - get_stepping_numbers(str(low_int-1))) % MOD
        return result