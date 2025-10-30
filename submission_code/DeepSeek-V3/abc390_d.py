def count_snake_numbers_up_to_n(n):
    s = str(n)
    length = len(s)
    total = 0
    
    # Count all snake numbers with fewer digits than 'length'
    for d in range(1, length):
        for k in range(1, 10):
            total += k ** (d - 1)
    
    # Now handle the numbers with 'length' digits
    first_digit = int(s[0])
    for k in range(1, first_digit):
        total += k ** (length - 1)
    
    # Now process the remaining digits
    remaining_length = length - 1
    current_max = first_digit
    valid = True
    for i in range(1, length):
        current_digit = int(s[i])
        remaining_digits = remaining_length
        # The digits from position i can be from 0 to min(current_max - 1, ...)
        # We can choose digits 0..(current_max - 1) for the remaining positions
        # But up to the current digit for the ith position
        upper = current_max - 1
        if upper < 0:
            valid = False
            break
        for d in range(0, current_digit):
            if d > upper:
                continue
            total += (upper + 1) ** (remaining_digits - 1)
        if current_digit >= current_max:
            valid = False
            break
        current_max = min(current_max, current_digit)
        remaining_length -= 1
    
    if valid:
        total += 1
    
    return total

L, R = map(int, input().split())

result = count_snake_numbers_up_to_n(R) - count_snake_numbers_up_to_n(L - 1)
print(result)