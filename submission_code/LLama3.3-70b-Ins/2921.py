class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        low, high = int(low), int(high)
        
        def is_stepping(n: int) -> bool:
            """Check if a number is a stepping number."""
            prev_digit = None
            while n:
                curr_digit = n % 10
                if prev_digit is not None and abs(prev_digit - curr_digit) != 1:
                    return False
                prev_digit = curr_digit
                n //= 10
            return True
        
        count = 0
        for num in range(low, high + 1):
            if is_stepping(num):
                count += 1
        return count % MOD