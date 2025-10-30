def count_snake_numbers(limit_str, is_upper_bound=True):
    n = len(limit_str)
    limit_digits = [int(d) for d in limit_str]
    
    # dp[pos][tight][started][first_digit] = count
    # pos: current position
    # tight: whether we're still bounded by the limit
    # started: whether we've placed the first non-zero digit
    # first_digit: the first digit we placed (or -1 if not started)
    
    memo = {}
    
    def dp(pos, tight, started, first_digit):
        if pos == n:
            return 1 if started else 0
        
        if (pos, tight, started, first_digit) in memo:
            return memo[(pos, tight, started, first_digit)]
        
        max_digit = limit_digits[pos] if tight else 9
        result = 0
        
        for digit in range(0, max_digit + 1):
            new_tight = tight and (digit == max_digit)
            new_started = started or (digit > 0)
            new_first_digit = first_digit
            
            # If we haven't started and this digit is 0, continue without starting
            if not started and digit == 0:
                result += dp(pos + 1, new_tight, False, -1)
            # If we haven't started and this digit > 0, this becomes our first digit
            elif not started and digit > 0:
                new_first_digit = digit
                result += dp(pos + 1, new_tight, True, new_first_digit)
            # If we have started, check if current digit is valid (< first_digit)
            elif started:
                if digit < first_digit:
                    result += dp(pos + 1, new_tight, True, new_first_digit)
        
        memo[(pos, tight, started, first_digit)] = result
        return result
    
    return dp(0, True, False, -1)

def count_snake_in_range(L, R):
    # Count snake numbers <= R
    count_R = count_snake_numbers(str(R), True)
    
    # Count snake numbers <= L-1
    if L > 0:
        count_L_minus_1 = count_snake_numbers(str(L - 1), True)
    else:
        count_L_minus_1 = 0
    
    return count_R - count_L_minus_1

# Read input
L, R = map(int, input().split())

# Calculate and print result
result = count_snake_in_range(L, R)
print(result)