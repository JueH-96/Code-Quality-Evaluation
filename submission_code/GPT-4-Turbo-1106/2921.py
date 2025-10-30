class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        low, high = int(low), int(high)
        
        # Helper function to count stepping numbers starting with 'num' and having 'n' digits
        def countStepping(num, n):
            if n == 0:
                return 1 if low <= num <= high else 0
            last_digit = num % 10
            count = 0
            # Append next digit if it's a stepping number
            if last_digit > 0:
                count += countStepping(num * 10 + last_digit - 1, n - 1)
            if last_digit < 9:
                count += countStepping(num * 10 + last_digit + 1, n - 1)
            return count % MOD
        
        # Count stepping numbers for each possible number of digits
        count = 0
        for num_digits in range(len(str(low)), len(str(high)) + 1):
            if num_digits == 1:
                # Single digit numbers are all stepping numbers
                count += min(high, 9) - max(low, 0) + 1
            else:
                for first_digit in range(1, 10):
                    count += countStepping(first_digit, num_digits - 1)
            count %= MOD
        
        return count

# Example usage:
# sol = Solution()
# print(sol.countSteppingNumbers("1", "11"))  # Output: 10
# print(sol.countSteppingNumbers("90", "101"))  # Output: 2