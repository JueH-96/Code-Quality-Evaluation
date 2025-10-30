class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        # Ensure num1 and num2 have the same length for easier processing
        num1 = num1.zfill(len(num2))
        mod = 10**9 + 7
        
        # Memoization cache
        memo = {}
        
        def dp(pos, tight_lower, tight_upper, curr_sum):
            # Base case: we've processed all digits
            if pos == len(num2):
                return 1 if min_sum <= curr_sum <= max_sum else 0
            
            # Return cached result if available
            if (pos, tight_lower, tight_upper, curr_sum) in memo:
                return memo[(pos, tight_lower, tight_upper, curr_sum)]
            
            # Early termination checks
            if curr_sum > max_sum:
                return 0
            
            # If even adding all 9's won't reach min_sum, return 0
            if curr_sum + 9 * (len(num2) - pos) < min_sum:
                return 0
            
            result = 0
            
            # Determine the range of valid digits for this position
            low_digit = int(num1[pos]) if tight_lower else 0
            high_digit = int(num2[pos]) if tight_upper else 9
            
            for digit in range(low_digit, high_digit + 1):
                # If adding this digit would exceed max_sum, skip
                if curr_sum + digit > max_sum:
                    continue
                
                # Update tight constraints for the next position
                next_tight_lower = tight_lower and digit == low_digit
                next_tight_upper = tight_upper and digit == high_digit
                
                result = (result + dp(pos + 1, next_tight_lower, next_tight_upper, curr_sum + digit)) % mod
            
            # Cache the result
            memo[(pos, tight_lower, tight_upper, curr_sum)] = result
            return result
        
        return dp(0, True, True, 0)