def count_snake_up_to(n):
    if n < 10:
        return 0
    
    str_n = str(n)
    len_n = len(str_n)
    count = 0
    
    # Count Snake numbers with fewer digits than n
    for length in range(2, len_n):
        for top_digit in range(1, 10):
            count += top_digit ** (length - 1)
    
    # Count Snake numbers with the same number of digits as n
    top_digit_n = int(str_n[0])
    
    for top_digit in range(1, top_digit_n):
        count += top_digit ** (len_n - 1)
    
    # Check if n's top digit can lead to Snake numbers
    if all(int(d) < top_digit_n for d in str_n[1:]):
        count += count_snake_with_prefix(top_digit_n, str_n[1:])
    
    return count

def count_snake_with_prefix(top_digit, remaining_digits):
    if not remaining_digits:
        return 1
    
    first_digit = int(remaining_digits[0])
    rest_of_digits = remaining_digits[1:]
    count = 0
    
    # Count all valid numbers with a digit less than the first digit
    for d in range(first_digit):
        if d < top_digit:
            count += top_digit ** len(rest_of_digits)
    
    # If the first digit is less than the top digit, continue counting
    if first_digit < top_digit:
        count += count_snake_with_prefix(top_digit, rest_of_digits)
    
    return count

def count_snake_numbers(L, R):
    return count_snake_up_to(R) - count_snake_up_to(L - 1)

# Read input
L, R = map(int, input().split())

# Calculate the answer
answer = count_snake_numbers(L, R)

# Print the answer
print(answer)