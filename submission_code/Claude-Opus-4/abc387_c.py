def count_snake_numbers(n):
    if n < 10:
        return 0
    
    s = str(n)
    digits = len(s)
    
    # Count Snake numbers with fewer digits than n
    total = 0
    for d in range(2, digits):
        # For d-digit numbers, first digit can be 1-9
        # All other digits must be strictly less than the first digit
        for first in range(1, 10):
            # Each of the remaining d-1 positions can have digits 0 to first-1
            total += first ** (d - 1)
    
    # Count Snake numbers with same number of digits as n
    # Use digit DP
    memo = {}
    
    def dp(pos, first_digit, is_limit, started):
        if pos == digits:
            return 1 if started else 0
        
        state = (pos, first_digit, is_limit, started)
        if state in memo:
            return memo[state]
        
        result = 0
        max_digit = int(s[pos]) if is_limit else 9
        
        for digit in range(0, max_digit + 1):
            if not started:
                # Haven't started the number yet
                if digit == 0:
                    # Still leading zeros
                    result += dp(pos + 1, -1, False, False)
                else:
                    # This is the first digit
                    result += dp(pos + 1, digit, is_limit and digit == max_digit, True)
            else:
                # Already started, check if digit is less than first_digit
                if digit < first_digit:
                    result += dp(pos + 1, first_digit, is_limit and digit == max_digit, True)
        
        memo[state] = result
        return result
    
    # Start DP from position 0
    total += dp(0, -1, True, False)
    
    return total

# Read input
L, R = map(int, input().split())

# Calculate answer
answer = count_snake_numbers(R) - count_snake_numbers(L - 1)
print(answer)