def find_nth_good_integer(N):
    # Determine the number of digits required
    digit_count = 1
    cumulative_count = 0
    
    while True:
        good_integers_count = 5 if digit_count == 1 else 4 * (5 ** (digit_count - 1))
        if cumulative_count + good_integers_count >= N:
            break
        cumulative_count += good_integers_count
        digit_count += 1
    
    # Adjust N to represent the position within the current digit category
    N = N - cumulative_count
    
    # Generate the good integer
    if digit_count == 1:
        # Handle the single-digit case
        return [0, 2, 4, 6, 8][N - 1]
    
    # Handle the multiple-digit case
    result = []
    
    # First digit (can't be 0)
    first_digit_options = [2, 4, 6, 8]
    idx = (N - 1) // (5 ** (digit_count - 1))
    result.append(str(first_digit_options[idx]))
    
    # Update N for the next digit
    N = (N - 1) % (5 ** (digit_count - 1)) + 1
    
    # Remaining digits
    for i in range(1, digit_count):
        digit_options = [0, 2, 4, 6, 8]
        idx = (N - 1) // (5 ** (digit_count - 1 - i))
        result.append(str(digit_options[idx]))
        
        # Update N for the next digit
        N = (N - 1) % (5 ** (digit_count - 1 - i)) + 1
    
    return int(''.join(result))

# Read N from standard input
N = int(input().strip())

# Print the N-th smallest good integer
print(find_nth_good_integer(N))