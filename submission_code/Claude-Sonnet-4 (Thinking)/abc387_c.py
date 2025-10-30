L, R = map(int, input().split())

def count_snake_numbers(limit):
    if limit < 10:
        return 0
    
    s = str(limit)
    n = len(s)
    
    memo = {}
    
    def dp(pos, tight, started, first_digit):
        if pos == n:
            return 1 if started else 0
        
        key = (pos, tight, started, first_digit)
        if key in memo:
            return memo[key]
        
        max_digit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, max_digit + 1):
            new_tight = tight and (digit == max_digit)
            
            if not started:
                if digit == 0:
                    result += dp(pos + 1, new_tight, False, -1)
                else:
                    result += dp(pos + 1, new_tight, True, digit)
            else:
                if digit < first_digit:
                    result += dp(pos + 1, new_tight, True, first_digit)
        
        memo[key] = result
        return result
    
    return dp(0, True, False, -1)

result = count_snake_numbers(R) - count_snake_numbers(L - 1)
print(result)