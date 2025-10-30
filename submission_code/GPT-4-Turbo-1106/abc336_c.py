def find_nth_good_integer(N):
    # The good integers are essentially numbers in base 5, where the digits are 0, 2, 4, 6, 8.
    # We can find the N-th good integer by converting N to base 5 and mapping the digits to 0, 2, 4, 6, 8.
    base_5 = []
    while N > 0:
        N, remainder = divmod(N - 1, 5)
        base_5.append(remainder)
    base_5.reverse()
    
    # Map the base 5 digits to the good integer digits
    good_integer_digits = {0: '0', 1: '2', 2: '4', 3: '6', 4: '8'}
    good_integer = ''.join(good_integer_digits[digit] for digit in base_5)
    
    return int(good_integer) if good_integer else 0

# Read input from stdin
N = int(input().strip())

# Find and print the N-th good integer
print(find_nth_good_integer(N))