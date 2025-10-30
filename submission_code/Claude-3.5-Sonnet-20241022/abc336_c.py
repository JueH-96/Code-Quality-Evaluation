N = int(input())

def get_nth_good_integer(n):
    # Handle 0 as a special case
    if n == 1:
        return 0
    
    # Subtract 1 from n since we handled 0
    n -= 1
    
    # Find length of number needed
    length = 1
    count = 5  # Count of 1-digit good numbers (2,4,6,8)
    while n > count:
        n -= count
        length += 1
        count *= 5

    # Generate the number digit by digit
    result = []
    even_digits = [0, 2, 4, 6, 8]
    
    # First digit can't be 0
    first_digit_idx = (n-1) // (5 ** (length-1))
    result.append(even_digits[first_digit_idx + 1])  # Skip 0 for first digit
    n = (n-1) % (5 ** (length-1)) + 1

    # Generate remaining digits
    for i in range(length-1):
        digit_idx = (n-1) // (5 ** (length-2-i))
        result.append(even_digits[digit_idx])
        n = (n-1) % (5 ** (length-2-i)) + 1

    # Convert to integer
    return int(''.join(map(str, result)))

print(get_nth_good_integer(N))