def count_snake_numbers_up_to(N):
    if N < 10:
        return 0
    
    s = str(N)
    n = len(s)
    
    # Count all Snake numbers with fewer digits than N
    count = 0
    for num_digits in range(2, n):
        for first_digit in range(1, 10):
            count += first_digit ** (num_digits - 1)
    
    # Count Snake numbers with exactly n digits that are <= N
    for first_digit in range(1, 10):
        if first_digit > int(s[0]):
            break
        elif first_digit < int(s[0]):
            count += first_digit ** (n - 1)
        else:
            # first_digit == int(s[0])
            count += count_with_first_digit_equal(s, first_digit)
    
    return count

def count_with_first_digit_equal(s, first_digit):
    n = len(s)
    
    def helper(pos):
        if pos == n:
            return 1
        
        result = 0
        limit = int(s[pos])
        
        for digit in range(0, min(first_digit, limit + 1)):
            if digit < limit:
                result += first_digit ** (n - pos - 1)
            else:
                # digit == limit
                result += helper(pos + 1)
        
        return result
    
    return helper(1)

L, R = map(int, input().split())
answer = count_snake_numbers_up_to(R) - count_snake_numbers_up_to(L - 1)
print(answer)